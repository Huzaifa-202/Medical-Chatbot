"""
Multilingual Pre-Sales Voicebot - Azure OpenAI Realtime API
============================================================
A real-time voice assistant optimized for pre-sales conversations with:
- Multilingual support (English, Urdu, Arabic, Spanish)
- Ultra-low latency voice interactions
- Intelligent interruption handling
- Automatic language detection and switching
- Comprehensive latency tracking

Architecture:
- WebSocket Layer: Handles Azure OpenAI Realtime API communication
- Audio Layer: Manages microphone input and speaker output (24kHz PCM16)
- Language Layer: Detects and switches languages dynamically
- Event Layer: Processes all API events (STT, TTS, responses, interruptions)

Author: Built with Claude Code
Version: 2.0 (Clean rewrite)
"""

import asyncio
import websockets
import json
import base64
import pyaudio
import os
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
import logging
import re
import time
from typing import Optional, Dict, List

# ============================================================
# CONFIGURATION
# ============================================================

# Make language detection deterministic (same input = same output)
DetectorFactory.seed = 0

# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


# ============================================================
# MULTILINGUAL VOICEBOT CLASS
# ============================================================

class MultilingualVoiceBot:
    """
    Main voicebot class handling all aspects of real-time voice conversation
    """

    def __init__(self):
        """Initialize voicebot with Azure OpenAI and audio configuration"""

        # ========================================
        # Azure OpenAI Configuration
        # ========================================
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        # Validate required environment variables
        if not all([self.api_key, self.endpoint, self.deployment]):
            raise ValueError(
                "Missing required environment variables. Please check your .env file:\n"
                "- OPENAI_API_KEY\n"
                "- AZURE_OPENAI_ENDPOINT\n"
                "- AZURE_OPENAI_DEPLOYMENT_NAME"
            )

        # ========================================
        # Audio Configuration (24kHz PCM16)
        # ========================================
        self.CHUNK_SIZE = 1024          # Audio chunk size in bytes
        self.AUDIO_FORMAT = pyaudio.paInt16  # 16-bit PCM audio
        self.CHANNELS = 1                # Mono audio
        self.SAMPLE_RATE = 24000         # 24kHz required by GPT-4o Realtime API

        self.audio = pyaudio.PyAudio()
        self.is_running = True

        # ========================================
        # Language & Conversation State
        # ========================================
        self.current_language = "en"  # Start with English
        self.conversation_history: List[Dict] = []

        # ========================================
        # Interruption Handling (CLIENT-SIDE)
        # ========================================
        # This flag enables INSTANT audio stopping when user speaks
        self.user_is_speaking = False

        # ========================================
        # Latency Tracking
        # ========================================
        self.speech_start_time: Optional[float] = None
        self.stt_complete_time: Optional[float] = None
        self.response_start_time: Optional[float] = None
        self.tts_start_time: Optional[float] = None

    # ============================================================
    # LANGUAGE DETECTION & MULTILINGUAL SUPPORT
    # ============================================================

    def is_english_text(self, text: str) -> bool:
        """
        Detect if text is English using keyword matching

        This prevents false detection where English with accent
        gets misclassified as Urdu/Hindi by langdetect

        Args:
            text: Input text to check

        Returns:
            True if text contains English indicators
        """
        # Common English words and patterns
        english_patterns = [
            # Common verbs
            r'\b(is|are|was|were|have|has|had|do|does|did)\b',
            r'\b(will|would|can|could|should|may|might)\b',
            # Pronouns
            r'\b(I|you|he|she|it|we|they|me|him|her|us|them)\b',
            # Question words
            r'\b(what|where|when|why|how|who|which)\b',
            # Common words
            r'\b(the|a|an|this|that|these|those)\b',
            r'\b(yes|no|hello|hi|thanks|please|sorry)\b',
            # Common verbs
            r'\b(want|need|like|get|make|know|think|see)\b'
        ]

        text_lower = text.lower()

        # Check for English keyword matches
        for pattern in english_patterns:
            if re.search(pattern, text_lower):
                return True

        # Check if mostly Latin alphabet (English uses Latin characters)
        latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        total_alpha = sum(1 for c in text if c.isalpha())

        if total_alpha > 0 and (latin_chars / total_alpha) > 0.7:
            return True

        return False

    def detect_language(self, text: str) -> str:
        """
        Detect language from text using hybrid approach

        Strategy:
        1. First check for English keywords (prevents false Urdu detection)
        2. Then use langdetect for other languages
        3. Map to supported languages

        Args:
            text: User's spoken text (transcribed)

        Returns:
            Language code: 'en', 'ur', 'ar', 'es'
        """
        try:
            # Skip very short text (not enough context)
            if len(text.split()) < 2:
                return self.current_language

            # STEP 1: Check for English first (keyword-based)
            if self.is_english_text(text):
                logger.info(f"🌍 Language: ENGLISH (keyword match) - '{text[:50]}...'")
                return "en"

            # STEP 2: Use langdetect for non-English
            detected = detect(text)
            logger.info(f"🌍 Language: {detected.upper()} - '{text[:50]}...'")

            # STEP 3: Map to supported languages
            language_map = {
                "en": "en",
                "ur": "ur",
                "hi": "ur",  # Hindi → use Urdu prompt (similar language)
                "ar": "ar",
                "es": "es",
                "fr": "en",  # French → fallback to English
                "de": "en",  # German → fallback to English
            }

            return language_map.get(detected, "en")

        except Exception as e:
            logger.warning(f"⚠️  Language detection failed: {e}")
            return self.current_language  # Keep current language on error

    def get_system_prompt(self, language: str = "en") -> str:
        """
        Get pre-sales optimized system prompt in specified language

        These prompts guide the AI to act as a professional pre-sales consultant
        who understands customer needs and presents solutions effectively.

        Args:
            language: Language code ('en', 'ur', 'ar', 'es')

        Returns:
            System prompt string in requested language
        """
        prompts = {
            "en": """You are an expert pre-sales consultant. Your role:

1. Understand customer needs and pain points deeply
2. Present relevant product/service benefits clearly
3. Handle objections professionally and empathetically
4. Guide conversation toward qualified opportunities
5. Ask smart qualifying questions (budget, timeline, decision-makers)
6. Be conversational, warm, and solution-focused

IMPORTANT:
- Keep responses concise (2-4 sentences) but COMPLETE
- ALWAYS respond in the SAME language the customer speaks
- DO NOT stop mid-sentence
- Listen actively and adapt to customer's communication style""",

            "ur": """Aap ek expert pre-sales consultant hain. Aapka kaam:

1. Customer ki zarooriyat aur mushkilat ko gehrai se samjhna
2. Product/service ke fawaid saaf tor par pesh karna
3. Etirazat ko professionally aur hamdardi se handle karna
4. Conversation ko qualified opportunities ki taraf le jana
5. Budget, timeline, aur decision-makers ke bare mein samajhdari se sawal karna
6. Baat cheet mein dosti aur hal-focused hona

ZAROORI:
- Jawab mukhtasar (2-4 jumlay) magar MUKAMMAL rakhen
- Hamesha customer ki ZUBAN mein jawab dein
- Beech mein kabhi na rukein
- Dhyan se sunein aur customer ke style ke mutabiq baat karein""",

            "ar": """أنت مستشار مبيعات خبير. دورك:

1. فهم احتياجات العميل ونقاط الألم بعمق
2. تقديم فوائد المنتج/الخدمة بوضوح
3. التعامل مع الاعتراضات بشكل احترافي وتعاطفي
4. توجيه المحادثة نحو الفرص المؤهلة
5. طرح أسئلة تأهيل ذكية (الميزانية، الجدول الزمني، صناع القرار)
6. كن محاوراً ودوداً ومركزاً على الحلول

مهم:
- اجعل الردود موجزة (2-4 جمل) لكن كاملة
- رد دائماً بنفس اللغة التي يتحدث بها العميل
- لا تتوقف في منتصف الجملة
- استمع بنشاط وتكيف مع أسلوب تواصل العميل""",

            "es": """Eres un consultor de preventas experto. Tu rol:

1. Comprender profundamente las necesidades y puntos de dolor del cliente
2. Presentar beneficios relevantes del producto/servicio claramente
3. Manejar objeciones profesional y empáticamente
4. Guiar la conversación hacia oportunidades calificadas
5. Hacer preguntas calificadoras inteligentes (presupuesto, cronograma, tomadores de decisiones)
6. Ser conversacional, cálido y enfocado en soluciones

IMPORTANTE:
- Mantén respuestas concisas (2-4 oraciones) pero COMPLETAS
- SIEMPRE responde en el MISMO idioma que habla el cliente
- NO te detengas a mitad de frase
- Escucha activamente y adáptate al estilo de comunicación del cliente"""
        }

        return prompts.get(language, prompts["en"])

    # ============================================================
    # WEBSOCKET CONNECTION & SESSION MANAGEMENT
    # ============================================================

    async def connect_to_realtime_api(self):
        """
        Establish WebSocket connection to Azure OpenAI Realtime API

        Azure OpenAI uses a different URL format than standard OpenAI:
        - WSS protocol (secure WebSocket)
        - Includes deployment name in URL
        - Uses 'api-key' header instead of 'Authorization'

        Returns:
            Connected WebSocket object
        """
        # Clean and format endpoint URL
        endpoint = self.endpoint.rstrip('/').replace('https://', '').replace('http://', '')

        # Construct WebSocket URL
        url = f"wss://{endpoint}/openai/realtime?api-version=2024-10-01-preview&deployment={self.deployment}"

        # Azure-specific headers
        headers = {
            "api-key": self.api_key
        }

        logger.info(f"🔗 Connecting to Azure OpenAI Realtime API...")
        logger.info(f"📍 Deployment: {self.deployment}")

        # Connect to WebSocket
        websocket = await websockets.connect(url, additional_headers=headers)
        logger.info("✅ Connected successfully!")

        return websocket

    async def configure_session(self, websocket) -> None:
        """
        Configure the Realtime API session with our settings

        This sets up:
        - Voice (shimmer)
        - Audio format (24kHz PCM16)
        - Voice Activity Detection (VAD) for interruption
        - System instructions (pre-sales consultant)
        - Transcription model (Whisper)

        Args:
            websocket: Connected WebSocket
        """
        config = {
            "type": "session.update",
            "session": {
                # Modalities: We want both text and audio
                "modalities": ["text", "audio"],

                # System instructions (pre-sales prompt in current language)
                "instructions": self.get_system_prompt(self.current_language),

                # Voice selection: shimmer (most natural sounding)
                "voice": "shimmer",

                # Audio formats: PCM16 at 24kHz (required by API)
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",

                # Enable transcription with Whisper
                "input_audio_transcription": {
                    "model": "whisper-1"
                },

                # Voice Activity Detection (VAD) - CRITICAL for interruption
                "turn_detection": {
                    "type": "server_vad",  # Server-side VAD
                    "threshold": 0.5,      # Sensitivity (0.0-1.0)
                    "prefix_padding_ms": 300,    # Audio before speech starts
                    "silence_duration_ms": 700,  # Silence duration to end turn
                    "create_response": False     # Manual response trigger (for language detection)
                },

                # Response generation settings
                "temperature": 0.8,  # Creativity level (0.0-1.0)
                "max_response_output_tokens": 1000  # Allow complete responses
            }
        }

        await websocket.send(json.dumps(config))
        logger.info("⚙️  Session configured successfully")

    async def update_language(self, websocket, new_language: str) -> None:
        """
        Update session language dynamically during conversation

        When language changes, we need to update the system prompt
        so the AI responds in the new language.

        Args:
            websocket: Connected WebSocket
            new_language: New language code
        """
        if new_language != self.current_language:
            logger.info(f"🌍 LANGUAGE SWITCH: {self.current_language} → {new_language}")
            self.current_language = new_language

            # Update session with new language prompt
            update = {
                "type": "session.update",
                "session": {
                    "instructions": self.get_system_prompt(new_language)
                }
            }

            await websocket.send(json.dumps(update))

    # ============================================================
    # AUDIO STREAMING
    # ============================================================

    async def stream_microphone_audio(self, websocket) -> None:
        """
        Capture audio from microphone and stream to API

        This function:
        1. Opens microphone stream
        2. Continuously reads audio chunks
        3. Encodes as base64
        4. Sends to API for transcription

        Args:
            websocket: Connected WebSocket
        """
        # Open microphone stream
        stream = self.audio.open(
            format=self.AUDIO_FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE
        )

        logger.info("🎤 Microphone active - Start speaking!")

        try:
            while self.is_running:
                # Read audio from microphone
                audio_bytes = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)

                # Encode to base64 for transmission
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

                # Send to API
                message = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_base64
                }

                await websocket.send(json.dumps(message))
                await asyncio.sleep(0.01)  # Prevent overwhelming the API

        except Exception as e:
            logger.error(f"❌ Microphone error: {e}")
        finally:
            stream.stop_stream()
            stream.close()

    # ============================================================
    # EVENT PROCESSING & AUDIO PLAYBACK
    # ============================================================

    async def process_events(self, websocket) -> None:
        """
        Process all events from the Realtime API

        This is the main event loop that handles:
        - Speech transcription (STT)
        - Response generation
        - Audio playback (TTS)
        - Interruption events
        - Latency tracking

        Args:
            websocket: Connected WebSocket
        """
        # Open speaker stream for audio playback
        speaker_stream = self.audio.open(
            format=self.AUDIO_FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            output=True,
            frames_per_buffer=self.CHUNK_SIZE * 2  # Larger buffer for smooth playback
        )

        try:
            async for message in websocket:
                data = json.loads(message)
                event_type = data.get("type")

                # ==========================================
                # AUDIO PLAYBACK (TTS)
                # ==========================================
                if event_type == "response.audio.delta":
                    # Received audio chunk from bot
                    audio_b64 = data.get("delta", "")
                    if audio_b64:
                        # Track TTS start time (first audio chunk)
                        if self.tts_start_time is None and self.response_start_time:
                            self.tts_start_time = time.time()
                            tts_latency = (self.tts_start_time - self.response_start_time) * 1000
                            logger.info(f"⏱️  TTS Latency: {tts_latency:.0f}ms")

                        # Decode audio
                        audio_bytes = base64.b64decode(audio_b64)

                        # ⚡ CLIENT-SIDE INTERRUPTION: Only play if user is NOT speaking
                        if not self.user_is_speaking:
                            speaker_stream.write(audio_bytes)
                        # Otherwise, discard audio chunk (instant stop!)

                # ==========================================
                # RESPONSE TEXT (for display/logging)
                # ==========================================
                elif event_type == "response.audio_transcript.delta":
                    # Bot's response text (streaming)
                    text = data.get("delta", "")
                    if text:
                        # Track response generation start
                        if self.response_start_time is None and self.stt_complete_time:
                            self.response_start_time = time.time()
                            resp_latency = (self.response_start_time - self.stt_complete_time) * 1000
                            logger.info(f"⏱️  Response Generation Latency: {resp_latency:.0f}ms")

                        # Print response as it streams
                        print(f"{text}", end="", flush=True)

                elif event_type == "response.audio_transcript.done":
                    # Bot finished speaking (complete response)
                    print()  # New line
                    transcript = data.get("transcript", "")
                    if transcript:
                        logger.info(f"🤖 Bot: {transcript}")
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": transcript
                        })

                        # Calculate end-to-end latency
                        if self.speech_start_time and self.tts_start_time:
                            e2e = (self.tts_start_time - self.speech_start_time) * 1000
                            logger.info(f"⏱️  📊 END-TO-END LATENCY: {e2e:.0f}ms")

                        print("-" * 70)

                        # Reset latency trackers
                        self.reset_latency_trackers()

                # ==========================================
                # SPEECH TRANSCRIPTION (STT)
                # ==========================================
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    # User's speech was transcribed
                    transcript = data.get("transcript", "")
                    if transcript:
                        # Track STT completion time
                        self.stt_complete_time = time.time()

                        # Calculate STT latency
                        if self.speech_start_time:
                            stt_latency = (self.stt_complete_time - self.speech_start_time) * 1000
                            logger.info(f"⏱️  STT Latency: {stt_latency:.0f}ms")

                        logger.info(f"👤 User: {transcript}")
                        self.conversation_history.append({
                            "role": "user",
                            "content": transcript
                        })

                        # Detect and switch language if needed
                        detected_lang = self.detect_language(transcript)
                        await self.update_language(websocket, detected_lang)

                        # NOW trigger response (after language is set correctly)
                        response_trigger = {"type": "response.create"}
                        await websocket.send(json.dumps(response_trigger))

                # ==========================================
                # INTERRUPTION HANDLING
                # ==========================================
                elif event_type == "input_audio_buffer.speech_started":
                    # ⚡ User started speaking - STOP PLAYBACK IMMEDIATELY
                    self.user_is_speaking = True
                    self.speech_start_time = time.time()

                    logger.warning("🎤 USER SPEAKING - STOPPING BOT")

                    # Send cancel to server (stops response generation)
                    cancel_msg = {"type": "response.cancel"}
                    await websocket.send(json.dumps(cancel_msg))

                elif event_type == "input_audio_buffer.speech_stopped":
                    # User stopped speaking - allow bot to respond
                    self.user_is_speaking = False
                    logger.info("🛑 User stopped speaking")

                elif event_type == "response.cancelled":
                    # Response was successfully cancelled
                    self.user_is_speaking = False
                    logger.warning("⚠️  Response cancelled (interrupted)")
                    print("\n[Interrupted]\n")

                # ==========================================
                # OTHER EVENTS
                # ==========================================
                elif event_type == "error":
                    error = data.get("error", {})
                    logger.error(f"❌ API Error: {error}")

                elif event_type == "session.updated":
                    logger.info("✅ Session updated")

                elif event_type == "response.done":
                    logger.info("✅ Response completed")

        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 Connection closed")
        except Exception as e:
            logger.error(f"❌ Event processing error: {e}")
        finally:
            speaker_stream.stop_stream()
            speaker_stream.close()

    # ============================================================
    # HELPER METHODS
    # ============================================================

    def reset_latency_trackers(self) -> None:
        """Reset all latency tracking variables for next interaction"""
        self.speech_start_time = None
        self.stt_complete_time = None
        self.response_start_time = None
        self.tts_start_time = None

    def cleanup(self) -> None:
        """Clean up resources on shutdown"""
        self.is_running = False
        if self.audio:
            self.audio.terminate()
        logger.info("🛑 Cleanup complete")

    # ============================================================
    # MAIN EXECUTION
    # ============================================================

    async def run(self) -> None:
        """
        Main execution method - runs the entire voicebot

        This method:
        1. Connects to Azure OpenAI Realtime API
        2. Configures the session
        3. Starts concurrent tasks for audio streaming and event processing
        4. Handles errors and cleanup
        """
        try:
            # Connect to API
            websocket = await self.connect_to_realtime_api()

            # Configure session
            await self.configure_session(websocket)

            # Start concurrent tasks
            mic_task = asyncio.create_task(self.stream_microphone_audio(websocket))
            event_task = asyncio.create_task(self.process_events(websocket))

            # Wait for both tasks
            await asyncio.gather(mic_task, event_task)

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        finally:
            self.cleanup()


# ============================================================
# ENTRY POINT
# ============================================================

async def main():
    """Application entry point"""
    print("\n" + "=" * 75)
    print("  🤖 MULTILINGUAL PRE-SALES VOICEBOT")
    print("=" * 75)
    print("  🎯 Purpose: Pre-sales conversations with intelligent language detection")
    print("  🗣️  Voice: Shimmer (Azure OpenAI)")
    print("  🌍 Languages: English, Urdu, Arabic, Spanish")
    print("  ⚡ Features:")
    print("     • Ultra-low latency voice interaction")
    print("     • Automatic language detection & switching")
    print("     • Client-side interruption (instant stop)")
    print("     • Real-time latency tracking (STT, Response, TTS, E2E)")
    print("=" * 75)

    # Display configuration
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "Not configured")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "Not configured")

    print(f"\n  📊 Model: {deployment}")
    print(f"  🔗 Endpoint: {endpoint[:60]}...")
    print("\n  💡 Tips:")
    print("     • Speak naturally in any supported language")
    print("     • Bot will automatically detect and respond in your language")
    print("     • Interrupt anytime - just start speaking!")
    print("     • Press Ctrl+C to exit")
    print("\n" + "=" * 75 + "\n")

    # Create and run bot
    bot = MultilingualVoiceBot()

    try:
        await bot.run()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        bot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

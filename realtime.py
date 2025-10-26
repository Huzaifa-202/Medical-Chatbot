"""
Real-time Multilingual Pre-Sales Voicebot
Uses OpenAI GPT-4o Realtime API for ultra-low latency voice conversations
Features: Auto language detection, interruption handling, pre-sales optimization
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
import traceback
import re
import time

# Make language detection deterministic
DetectorFactory.seed = 0

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PreSalesVoiceBot:
    def __init__(self):
        # Azure OpenAI Configuration
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not self.azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT not found in environment variables")
        if not self.deployment_name:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME not found in environment variables")

        # Audio configuration
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 24000  # GPT-4o Realtime API uses 24kHz

        self.audio = pyaudio.PyAudio()
        self.is_recording = True
        self.current_language = "en"

        # Pre-sales conversation context
        self.conversation_history = []

        # Language detection - MULTILINGUAL (responds in user's language)
        self.language_detection_buffer = []  # Store last 2 detections

        # Latency tracking
        self.speech_start_time = None
        self.transcription_complete_time = None
        self.response_start_time = None
        self.audio_playback_start_time = None

        # CLIENT-SIDE INTERRUPTION: Flag to immediately stop audio playback when user speaks
        self.user_is_speaking = False

    def get_presales_system_prompt(self, language="en"):
        """Generate pre-sales optimized system prompt in detected language"""
        prompts = {
            "en": """You are an expert pre-sales consultant for our company. Your role is to:
1. Understand customer needs and pain points
2. Present relevant product/service benefits
3. Handle objections professionally
4. Guide conversation towards qualified opportunities
5. Be conversational, empathetic, and solution-focused
6. Ask qualifying questions to understand budget, timeline, decision-makers

Keep responses concise but COMPLETE (2-4 sentences). Finish your thoughts.
ALWAYS respond in the SAME language the customer is speaking (English, Urdu, Arabic, etc.).
DO NOT stop mid-sentence. Complete your response.""",

            "ur": """Aap hamari company ke liye ek expert pre-sales consultant hain. Aapka role hai:
1. Customer ki zarooriyat aur problems ko samajhna
2. Mutalliq product/service ke fawaid pesh karna
3. Etirazat ko professionally handle karna
4. Baat cheet ko qualified opportunities ki taraf le jana
5. Baat cheet mein hamdardana aur hal par mabni hona
6. Budget, timeline, aur decision-makers ko samajhne ke liye sawalat poochna

Mukhtasar magar MUKAMMAL jawabat dein (2-4 jumlay). Apni baat puri karein.
Hamesha customer ki zuban mein jawab dein. Beech mein na rukein.""",

            "es": """Eres un consultor experto en preventas. Tu rol es:
1. Comprender las necesidades y puntos de dolor del cliente
2. Presentar beneficios relevantes del producto/servicio
3. Manejar objeciones profesionalmente
4. Guiar la conversación hacia oportunidades calificadas
5. Ser conversacional, empático y enfocado en soluciones

Mantén respuestas concisas pero COMPLETAS (2-4 oraciones). Termina tus pensamientos.
Siempre responde en el MISMO idioma que habla el cliente. NO pares a mitad de frase.""",

            "ar": """أنت مستشار مبيعات خبير لشركتنا. دورك هو:
1. فهم احتياجات العميل ونقاط الألم
2. تقديم فوائد المنتج/الخدمة ذات الصلة
3. التعامل مع الاعتراضات بشكل احترافي
4. توجيه المحادثة نحو الفرص المؤهلة
5. كن محاوراً متعاطفاً ومركزاً على الحلول
6. اطرح أسئلة مؤهلة لفهم الميزانية والجدول الزمني وصناع القرار

احتفظ بإجابات موجزة ولكن كاملة (2-4 جمل). أنهِ أفكارك.
رد دائماً بنفس اللغة التي يتحدث بها العميل. لا تتوقف في منتصف الجملة."""
        }

        return prompts.get(language, prompts["en"])

    def is_english_content(self, text):
        """
        Check if text contains English words/patterns to avoid false language detection
        This helps prevent detecting English with accent as Urdu/Hindi
        """
        # Common English words and patterns
        english_indicators = [
            r'\b(the|is|are|was|were|have|has|had|do|does|did|will|would|can|could|should|may|might)\b',
            r'\b(I|you|he|she|it|we|they|me|him|her|us|them)\b',
            r'\b(this|that|these|those|what|where|when|why|how|who)\b',
            r'\b(yes|no|hello|hi|thanks|thank you|please|sorry)\b',
            r'\b(want|need|like|make|get|go|come|see|know|think)\b',
            r'\b(good|bad|great|nice|fine|okay|ok)\b'
        ]

        text_lower = text.lower()

        # Check if any English patterns are found
        for pattern in english_indicators:
            if re.search(pattern, text_lower):
                return True

        # Check if mostly Latin characters (English alphabet)
        latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        total_chars = sum(1 for c in text if c.isalpha())

        if total_chars > 0 and (latin_chars / total_chars) > 0.7:
            return True

        return False

    async def connect_realtime_api(self):
        """Connect to Azure OpenAI Realtime API via WebSocket"""
        # Azure OpenAI Realtime API endpoint format
        # Remove trailing slash from endpoint if present
        endpoint = self.azure_endpoint.rstrip('/')
        # Remove https:// and construct WSS URL
        endpoint = endpoint.replace('https://', '').replace('http://', '')

        url = f"wss://{endpoint}/openai/realtime?api-version=2024-10-01-preview&deployment={self.deployment_name}"

        headers = {
            "api-key": self.api_key  # Azure uses 'api-key' header, not 'Authorization'
        }

        logger.info(f"Connecting to: {url}")
        return await websockets.connect(url, extra_headers=headers)

    async def initialize_session(self, websocket):
        """Initialize the session with configuration"""
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self.get_presales_system_prompt(self.current_language),
                "voice": "shimmer",  # Most natural/expressive voice (Options: alloy, echo, shimmer)
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",  # Voice Activity Detection - ENABLES INTERRUPTION
                    "threshold": 0.4,  # Lower = more sensitive = detects interruption faster
                    "prefix_padding_ms": 200,  # Reduced padding for faster interruption
                    "silence_duration_ms": 500,  # How long silence before considering speech done
                    "create_response": True  # Auto-create response when user stops speaking
                },
                "temperature": 0.8,
                "max_response_output_tokens": 1000  # Allow complete responses without cutting off
                # Note: voice_settings (speed) is not supported by Azure OpenAI Realtime API
            }
        }

        await websocket.send(json.dumps(session_config))
        logger.info("Session initialized with pre-sales configuration")

    async def detect_and_update_language(self, text, websocket):
        """
        MULTILINGUAL: Detects language accurately and switches to respond in user's language
        Uses English keyword detection first to avoid false Urdu detection
        """
        try:
            # Skip very short text (not enough context)
            if len(text.split()) < 2:
                return

            # STEP 1: Check if it's clearly English using keywords FIRST
            if self.is_english_content(text):
                detected_lang = "en"
                logger.info(f"🌍 Detected ENGLISH (keyword match): '{text}'")
            else:
                # STEP 2: Use langdetect for non-English languages
                detected_lang = detect(text)
                logger.info(f"🌍 Detected {detected_lang}: '{text}'")

            # Add to buffer for tracking
            self.language_detection_buffer.append(detected_lang)
            if len(self.language_detection_buffer) > 2:
                self.language_detection_buffer.pop(0)

            # Map detected language to supported languages
            lang_map = {
                "en": "en",
                "ur": "ur",
                "es": "es",
                "ar": "ar",
                "hi": "ur",  # Hindi -> use Urdu (Roman Urdu) prompt
                "fr": "en",
                "de": "en"
            }

            mapped_lang = lang_map.get(detected_lang, "en")

            # Switch if we have consistent detection
            if len(self.language_detection_buffer) >= 1:
                # Switch immediately for clear language changes
                if mapped_lang != self.current_language:
                    logger.info(f"🌍 LANGUAGE SWITCH: {self.current_language} → {mapped_lang}")
                    self.current_language = mapped_lang

                    # Update session with new language prompt
                    update_config = {
                        "type": "session.update",
                        "session": {
                            "instructions": self.get_presales_system_prompt(mapped_lang)
                        }
                    }
                    await websocket.send(json.dumps(update_config))

        except Exception as e:
            logger.warning(f"⚠️ Language detection failed: {e}")

    async def send_audio_stream(self, websocket):
        """
        STT INPUT: Stream audio from microphone to API
        This function captures your voice and sends it to OpenAI for transcription
        """
        # Open microphone stream to capture audio
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,  # Input = microphone recording
            frames_per_buffer=self.CHUNK
        )

        logger.info("🎤 Microphone started. Speak to the bot...")

        try:
            while self.is_recording:
                # STT STEP 1: Capture raw audio from microphone (1024 bytes at a time)
                audio_data = stream.read(self.CHUNK, exception_on_overflow=False)

                # STT STEP 2: Convert raw audio bytes to base64 format for transmission
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')

                # STT STEP 3: Package audio data into API message format
                message = {
                    "type": "input_audio_buffer.append",  # Tell API: "Here's more audio to transcribe"
                    "audio": audio_b64  # The actual audio data
                }

                # STT STEP 4: Send audio chunk to OpenAI API via WebSocket
                await websocket.send(json.dumps(message))
                await asyncio.sleep(0.01)  # Small delay to prevent overwhelming the API

        except Exception as e:
            logger.error(f"Error in audio stream: {e}")
        finally:
            stream.stop_stream()
            stream.close()

    async def receive_and_play_audio(self, websocket):
        """
        STT OUTPUT + TTS: Receive transcriptions and audio responses from API
        This function handles BOTH:
        1. Your speech transcription results (STT OUTPUT)
        2. Bot's voice responses (TTS)
        """
        # Open speaker stream to play bot's audio responses
        # Use larger buffer for smoother playback
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            output=True,  # Output = speaker playback
            frames_per_buffer=self.CHUNK * 2  # Double buffer size for smoother playback
        )

        try:
            # Listen for messages from OpenAI API
            async for message in websocket:
                data = json.loads(message)
                event_type = data.get("type")

                # ========== TTS: BOT SPEAKS ==========
                if event_type == "response.audio.delta":
                    # TTS STEP 1: Receive audio chunk from API (bot's voice)
                    audio_b64 = data.get("delta", "")
                    if audio_b64:
                        # Track first audio chunk (TTS start)
                        if self.audio_playback_start_time is None and self.response_start_time:
                            self.audio_playback_start_time = time.time()
                            tts_latency = (self.audio_playback_start_time - self.response_start_time) * 1000
                            logger.info(f"⏱️  TTS Latency: {tts_latency:.0f}ms")

                        # TTS STEP 2: Decode base64 audio to raw bytes
                        audio_data = base64.b64decode(audio_b64)

                        # ⚡ CLIENT-SIDE INTERRUPTION: Skip audio playback if user is speaking
                        if not self.user_is_speaking:
                            # TTS STEP 3: Play audio through speakers immediately
                            stream.write(audio_data)
                        # If user is speaking, discard this audio chunk (don't play it)

                elif event_type == "response.audio_transcript.delta":
                    # Bot's response text (for logging/display)
                    transcript = data.get("delta", "")
                    if transcript:
                        # Track first response text (Response generation complete)
                        if self.response_start_time is None and self.transcription_complete_time:
                            self.response_start_time = time.time()
                            response_latency = (self.response_start_time - self.transcription_complete_time) * 1000
                            logger.info(f"⏱️  Response Generation Latency: {response_latency:.0f}ms")
                        
                        print(f"> Bot: {transcript}", end="", flush=True)

                # ========== STT OUTPUT: YOUR SPEECH TRANSCRIBED ==========
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    # STT RESULT: Your voice has been converted to text!
                    transcript = data.get("transcript", "")
                    if transcript:
                        # Track STT completion time
                        self.transcription_complete_time = time.time()

                        # Calculate STT latency (from speech start to transcription)
                        if self.speech_start_time:
                            stt_latency = (self.transcription_complete_time - self.speech_start_time) * 1000
                            logger.info(f"⏱️  STT Latency: {stt_latency:.0f}ms")

                        logger.info(f"👤 User: {transcript}")  # Display what you said
                        self.conversation_history.append({"role": "user", "content": transcript})

                        # Auto-detect language from your speech and update bot's language
                        await self.detect_and_update_language(transcript, websocket)

                elif event_type == "response.audio_transcript.done":
                    # TTS COMPLETE: Bot finished speaking (get full transcript)
                    print()  # New line after bot response
                    transcript = data.get("transcript", "")
                    if transcript:
                        # Log complete response
                        logger.info(f"🤖 Bot (complete): {transcript}")
                        self.conversation_history.append({"role": "assistant", "content": transcript})

                        # Calculate end-to-end latency
                        if self.speech_start_time and self.audio_playback_start_time:
                            end_to_end = (self.audio_playback_start_time - self.speech_start_time) * 1000
                            logger.info(f"⏱️  📊 END-TO-END LATENCY: {end_to_end:.0f}ms")

                        # Print separator for readability
                        print("-" * 60)

                        # Reset timers for next interaction
                        self.speech_start_time = None
                        self.transcription_complete_time = None
                        self.response_start_time = None
                        self.audio_playback_start_time = None

                elif event_type == "conversation.item.input_audio_transcription.failed":
                    # STT ERROR: Failed to transcribe your speech
                    logger.warning("Transcription failed")

                elif event_type == "error":
                    logger.error(f"API Error: {data.get('error')}")

                elif event_type == "response.done":
                    logger.info("✅ Response completed successfully")

                elif event_type == "response.created":
                    logger.info("🔄 Response generation started")

                elif event_type == "response.output_item.added":
                    logger.info("📝 Response output item added")

                elif event_type == "input_audio_buffer.speech_started":
                    # ⚡ CRITICAL: Set flag to IMMEDIATELY stop audio playback on client side
                    self.user_is_speaking = True

                    # Track when user starts speaking (for latency measurement)
                    self.speech_start_time = time.time()
                    logger.warning("🎤 ⚠️  USER INTERRUPTING - STOPPING BOT NOW!")

                    # Send cancel command to stop server-side response generation
                    try:
                        cancel_message = {
                            "type": "response.cancel"
                        }
                        await websocket.send(json.dumps(cancel_message))
                        logger.info("✋ Sent cancel command to stop bot (server-side)")
                    except Exception as e:
                        logger.error(f"Failed to send cancel: {e}")

                elif event_type == "input_audio_buffer.speech_stopped":
                    # ⚡ Reset flag to allow bot audio playback again
                    self.user_is_speaking = False
                    logger.info("🛑 User stopped speaking - bot can respond now")

                elif event_type == "response.cancelled":
                    # Response successfully cancelled - ensure audio is stopped
                    self.user_is_speaking = False  # Reset for next response
                    logger.warning("⚠️  ✋ Response CANCELLED - User interrupted!")
                    print("\n[Response interrupted by user]\n")

                elif event_type == "input_audio_buffer.cleared":
                    logger.info("🔄 Audio buffer cleared")

                elif event_type == "input_audio_buffer.committed":
                    logger.info("✅ Audio buffer committed - ready to process")

                elif event_type == "conversation.item.truncated":
                    logger.warning("✂️  Conversation item truncated (interrupted)")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
        finally:
            stream.stop_stream()
            stream.close()

    async def run(self):
        """Main bot execution"""
        try:
            logger.info("Connecting to OpenAI Realtime API...")
            websocket = await self.connect_realtime_api()
            logger.info(" Connected successfully!")

            # Initialize session
            await self.initialize_session(websocket)

            # Create concurrent tasks for sending and receiving
            send_task = asyncio.create_task(self.send_audio_stream(websocket))
            receive_task = asyncio.create_task(self.receive_and_play_audio(websocket))

            # Wait for both tasks
            await asyncio.gather(send_task, receive_task)

        except Exception as e:
            logger.error(f"Error running bot: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.is_recording = False
        if self.audio:
            self.audio.terminate()
        logger.info("Bot shutdown complete")

async def main():
    """Entry point"""
    print("=" * 75)
    print("  🤖 Pre-Sales Voicebot - Multilingual Azure OpenAI Realtime API")
    print("=" * 75)
    print("  ✓ Ultra-low latency")
    print("  ✓ Multilingual (English ↔ Urdu ↔ Arabic ↔ Spanish)")
    print("  ✓ CLIENT-SIDE INTERRUPTION (instant stop when you speak)")
    print("  ✓ Real-time latency measurements")
    print("  ✓ Shimmer voice")
    print("=" * 75)
    print(f"\n  📊 Model: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'Not configured')}")
    print(f"  🔗 Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT', 'Not configured')[:50]}...")
    print("\n  🌍 Language Detection: English keywords checked FIRST")
    print("  🎤 Interrupt: Just start speaking while bot talks")
    print("  ⏱️  Latency: STT → Response → TTS → Total")
    print("\n  Press Ctrl+C to stop\n")
    print("=" * 75)

    bot = PreSalesVoiceBot()

    try:
        await bot.run()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down bot...")
        bot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

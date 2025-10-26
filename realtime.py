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
from langdetect import detect
import logging
import traceback

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
7. Adapt your language to match the customer's language automatically

Keep responses concise (2-3 sentences) for natural conversation flow.
Always respond in the SAME language the customer is speaking.""",

            "ur": """Aap hamari company ke liye ek expert pre-sales consultant hain. Aapka role hai:
1. Customer ki zarooriyat aur problems ko samajhna
2. Mutalliq product/service ke fawaid pesh karna
3. Etirazat ko professionally handle karna
4. Baat cheet ko qualified opportunities ki taraf le jana
5. Baat cheet mein hamdardana aur hal par mabni hona
6. Budget, timeline, aur decision-makers ko samajhne ke liye sawalat poochna

Qudrati baat cheet ki rawani ke liye mukhtasar jawabat dein (2-3 jumlay).
Hamesha usi zuban mein jawab dein jis mein customer baat kar raha hai.""",

            "es": """Eres un consultor experto en preventas. Tu rol es:
1. Comprender las necesidades y puntos de dolor del cliente
2. Presentar beneficios relevantes del producto/servicio
3. Manejar objeciones profesionalmente
4. Guiar la conversaci�n hacia oportunidades calificadas
5. Ser conversacional, emp�tico y enfocado en soluciones

Mant�n respuestas concisas (2-3 oraciones) para un flujo natural.
Siempre responde en el MISMO idioma que habla el cliente.""",

            "ar": """أنت مستشار مبيعات خبير لشركتنا. دورك هو:
1. فهم احتياجات العميل ونقاط الألم
2. تقديم فوائد المنتج/الخدمة ذات الصلة
3. التعامل مع الاعتراضات بشكل احترافي
4. توجيه المحادثة نحو الفرص المؤهلة
5. كن محاوراً متعاطفاً ومركزاً على الحلول
6. اطرح أسئلة مؤهلة لفهم الميزانية والجدول الزمني وصناع القرار

حافظ على إجابات موجزة (2-3 جمل) للتدفق الطبيعي للمحادثة.
رد دائماً بنفس اللغة التي يتحدث بها العميل."""
        }

        return prompts.get(language, prompts["en"])

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
                    "type": "server_vad",  # Voice Activity Detection for interruption
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },
                "temperature": 0.8,
                "max_response_output_tokens": 150  # Keep responses concise
            }
        }

        await websocket.send(json.dumps(session_config))
        logger.info("Session initialized with pre-sales configuration")

    async def detect_and_update_language(self, text, websocket):
        """Detect language and update system prompt if changed"""
        try:
            detected_lang = detect(text)

            # Map detected language to supported languages
            lang_map = {
                "en": "en",
                "ur": "ur",
                "es": "es",
                "ar": "ar",
                "hi": "en",  # Hindi -> use English prompt with multilingual capability
                "fr": "en",
                "de": "en"
            }

            mapped_lang = lang_map.get(detected_lang, "en")

            if mapped_lang != self.current_language:
                logger.info(f"Language changed from {self.current_language} to {mapped_lang}")
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
            logger.warning(f"Language detection failed: {e}")

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
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            output=True,  # Output = speaker playback
            frames_per_buffer=self.CHUNK
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
                        # TTS STEP 2: Decode base64 audio to raw bytes
                        audio_data = base64.b64decode(audio_b64)
                        # TTS STEP 3: Play audio through speakers immediately
                        stream.write(audio_data)

                elif event_type == "response.audio_transcript.delta":
                    # Bot's response text (for logging/display)
                    transcript = data.get("delta", "")
                    if transcript:
                        print(f"> Bot: {transcript}", end="", flush=True)

                # ========== STT OUTPUT: YOUR SPEECH TRANSCRIBED ==========
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    # STT RESULT: Your voice has been converted to text!
                    transcript = data.get("transcript", "")
                    if transcript:
                        logger.info(f"👤 User: {transcript}")  # Display what you said
                        self.conversation_history.append({"role": "user", "content": transcript})

                        # Auto-detect language from your speech and update bot's language
                        await self.detect_and_update_language(transcript, websocket)

                elif event_type == "response.audio_transcript.done":
                    # TTS COMPLETE: Bot finished speaking (get full transcript)
                    print()  # New line after bot response
                    transcript = data.get("transcript", "")
                    if transcript:
                        self.conversation_history.append({"role": "assistant", "content": transcript})

                elif event_type == "conversation.item.input_audio_transcription.failed":
                    # STT ERROR: Failed to transcribe your speech
                    logger.warning("Transcription failed")

                elif event_type == "error":
                    logger.error(f"API Error: {data.get('error')}")

                elif event_type == "response.done":
                    logger.info("Response completed")

                elif event_type == "input_audio_buffer.speech_started":
                    logger.info("🎤 User started speaking (interruption detected)")

                elif event_type == "input_audio_buffer.speech_stopped":
                    logger.info("🛑 User stopped speaking")

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
    print("=" * 60)
    print("> Pre-Sales Voicebot - Real-time Multilingual Assistant")
    print("=" * 60)
    print("Features:")
    print("   Ultra-low latency (~300ms)")
    print("   Automatic language detection & switching")
    print("   Interruption handling (bot stops when you speak)")
    print("   Pre-sales conversation optimization")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")

    bot = PreSalesVoiceBot()

    try:
        await bot.run()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down bot...")
        bot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

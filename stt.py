import azure.cognitiveservices.speech as speechsdk
import time, csv, datetime, statistics, atexit, os
from dotenv import load_dotenv

# 🌍 Load environment variables
load_dotenv()
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")

if not SPEECH_KEY or not SPEECH_REGION:
    raise ValueError("❌ Missing SPEECH_KEY or SPEECH_REGION in .env file!")

# 🎛️ Speech configuration
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.set_property(
    speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
    "Continuous"
)

# 🌐 Auto language detection
auto_detect_source_lang_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
    languages=["en-US", "ur-IN"]
)

# 🎙️ Input source
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

# 🪄 Recognizer
recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config,
    audio_config=audio_config,
    auto_detect_source_language_config=auto_detect_source_lang_config
)

# 💬 Phrase biasing
phrase_list = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
for phrase in ["hello", "thank you", "good morning", "Systems Limited", "please", "Deepgram", "Microsoft"]:
    phrase_list.addPhrase(phrase)
for phrase in ["آج", "کیا", "دن", "ہے", "کیا حال ہے", "شکریہ", "کیسے ہو"]:
    phrase_list.addPhrase(phrase)

# 🧾 CSV setup
filename = "stt_latency_results.csv"
csv_file = open(filename, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow(["Type", "Detected Language", "Recognized Text", "Latency (ms)", "Timestamp"])

# ⏱️ Tracking
utterance_start_time = None
chunk_latencies, utterance_summaries = [], []
first_chunk_latency = None
session_start = time.time()
last_detected_language = None
full_conversation = []

# 🕓 Silence tracking
last_speech_time = time.time()
SILENCE_TIMEOUT = 5  # seconds of silence before auto-stop

def recognizing_handler(evt):
    global utterance_start_time, first_chunk_latency, last_detected_language, last_speech_time
    if utterance_start_time is None:
        utterance_start_time = time.time()

    latency = (time.time() - utterance_start_time) * 1000
    if first_chunk_latency is None:
        first_chunk_latency = latency

    text = evt.result.text.strip()
    detected_lang = evt.result.properties.get(
        speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
    )

    if text:
        last_speech_time = time.time()
        chunk_latencies.append(latency)
        print(f"🟡 Partial ({detected_lang or 'Unknown'}): {text} | {latency:.2f} ms")
        writer.writerow(["Partial", detected_lang, text, f"{latency:.2f}", datetime.datetime.now()])

def recognized_handler(evt):
    global utterance_start_time, chunk_latencies, first_chunk_latency, last_detected_language, full_conversation, last_speech_time

    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        total_latency = (time.time() - utterance_start_time) * 1000
        text = evt.result.text.strip()
        detected_lang = evt.result.properties.get(
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
        )

        if text:
            full_conversation.append(text)
            last_speech_time = time.time()

            avg_latency = statistics.mean(chunk_latencies) if chunk_latencies else 0
            summary = {
                "detected_language": detected_lang or "Unknown",
                "total_speech_time_s": total_latency / 1000,
                "first_chunk_latency_ms": first_chunk_latency or 0,
                "avg_chunk_latency_ms": avg_latency,
                "final_latency_ms": total_latency
            }
            utterance_summaries.append(summary)

            print(f"\n✅ Final ({detected_lang}): {text}")
            print("📊 --- Utterance Summary ---")
            for k, v in summary.items():
                print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
            print("----------------------------\n")

            writer.writerow(["Final", detected_lang, text, f"{total_latency:.2f}", datetime.datetime.now()])

        utterance_start_time = None
        first_chunk_latency = None
        chunk_latencies.clear()
        last_detected_language = None

def session_summary():
    if not utterance_summaries:
        return
    avg_total_time = statistics.mean([s["total_speech_time_s"] for s in utterance_summaries])
    avg_first_latency = statistics.mean([s["first_chunk_latency_ms"] for s in utterance_summaries])
    avg_chunk_latency = statistics.mean([s["avg_chunk_latency_ms"] for s in utterance_summaries])
    avg_final_latency = statistics.mean([s["final_latency_ms"] for s in utterance_summaries])

    print("\n🧾 --- Session Summary ---")
    print(f"Average Speech Duration (s): {avg_total_time:.2f}")
    print(f"Average First Chunk Latency (ms): {avg_first_latency:.2f}")
    print(f"Average Chunk Latency (ms): {avg_chunk_latency:.2f}")
    print(f"Average Final Latency (ms): {avg_final_latency:.2f}")
    print(f"Total Session Time: {(time.time() - session_start):.2f}s")
    print("--------------------------")

    if full_conversation:
        full_text = " ".join(full_conversation)
        print("\n🗣️ --- Full Conversation Transcript ---")
        print(full_text)
        print("---------------------------------------")
        writer.writerow(["Full Conversation", "", full_text, "", datetime.datetime.now()])

    csv_file.close()

atexit.register(session_summary)

# 🔗 Handlers
recognizer.recognizing.connect(recognizing_handler)
recognizer.recognized.connect(recognized_handler)

print("🎙️ Speak in English or Urdu — Continuous detection enabled.")
print("🟢 Speak naturally. Auto-stops after 5 seconds of silence.\n")

recognizer.start_continuous_recognition()

try:
    while True:
        time.sleep(0.5)
        # 🕓 Check silence timeout
        if time.time() - last_speech_time > SILENCE_TIMEOUT:
            print("\n⏹️ Detected silence — stopping recognition automatically...")
            recognizer.stop_continuous_recognition()
            break
except KeyboardInterrupt:
    recognizer.stop_continuous_recognition()
    print("\n🛑 Recognition stopped manually.")

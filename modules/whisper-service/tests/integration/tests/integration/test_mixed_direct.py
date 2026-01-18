#!/usr/bin/env python3
"""Test mixed language audio directly with Whisper service"""

import sys

sys.path.insert(0, "../..")
import time

import librosa
import socketio
import soundfile as sf

print("🔌 Connecting to Whisper service at ws://localhost:5001")
sio = socketio.Client()
results = []


@sio.on("connect")
def on_connect():
    print("✅ Connected")


@sio.on("joined_session")
def on_joined(data):
    cs = data.get("config", {}).get("enable_code_switching")
    lang = data.get("config", {}).get("language")
    print(f"✅ Joined: code_switching={cs}, language={lang}")


@sio.on("transcription")
def on_transcription(data):
    text = data.get("text", "")
    language = data.get("language", "?")
    is_final = data.get("is_final", False)
    status = "✅ FINAL" if is_final else "⏳ PARTIAL"
    results.append(data)
    print(f"{status} | 🌐 {language.upper()} | {text[:80]}")


sio.connect("http://localhost:5001")

# Join with code-switching + auto language
config = {
    "model": "large-v3-turbo",
    "enable_code_switching": True,
    "language": "auto",
    "enable_vad": True,
}
sio.emit("join_session", {"session_id": "test-cs-auto", "config": config})
time.sleep(1)

# Load mixed audio
print("📁 Loading mixed audio")
audio, sr = sf.read("../fixtures/audio/test_clean_mixed_en_zh.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# Stream in 2s chunks
chunk_dur = 2.0
chunk_size = int(chunk_dur * 16000)
print(f"📦 Streaming {len(audio)/16000:.1f}s in {chunk_dur}s chunks")

for i in range(0, len(audio), chunk_size):
    chunk = audio[i : i + chunk_size]
    sio.emit(
        "transcribe_stream",
        {"session_id": "test-cs-auto", "audio_data": chunk.tobytes(), "sample_rate": 16000},
    )
    time.sleep(0.5)

print("⏳ Waiting for results...")
time.sleep(5)
sio.disconnect()

print(f"\n📊 Total results: {len(results)}")
for i, r in enumerate(results[:10]):
    text = r.get("text", "")[:60]
    lang = r.get("language", "?")
    print(f"  {i+1}. [{lang.upper()}] {text}")

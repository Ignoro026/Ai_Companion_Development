from flask import Flask, request, jsonify, send_file, url_for
import sounddevice as sd
import numpy as np
import tempfile
import wave
from faster_whisper import WhisperModel
import pyttsx3
import os
import uuid

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # 한글이 유니코드(\uXXXX)로 안 나오고 그대로 표시되게 함


# 저장할 폴더
OUTPUT_DIR = "tts_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Whisper 모델 로드
model = WhisperModel("small", device="cpu", compute_type="int8")

# pyttsx3 초기화
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 180)
tts_engine.setProperty("volume", 1.0)

def record_audio(duration=5, samplerate=16000):
    print("🎤 음성 입력 대기 중...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    print("✅ 녹음 완료")
    return np.squeeze(audio)

def transcribe_audio(audio, samplerate=16000):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        with wave.open(tmpfile.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio.tobytes())
        tmp_wav = tmpfile.name

    segments, info = model.transcribe(tmp_wav, language=None)
    text = " ".join([seg.text for seg in segments])
    print(f"🌍 감지된 언어: {info.language}")
    return text.strip(), info.language

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_text = data.get("text", "")
    lang = data.get("language", "en")

    # 간단한 규칙 기반 응답
    if lang == "ko":
        if "안녕" in user_text:
            reply = "안녕하세요! 만나서 반가워요."
            emotion = "joy"
        elif "슬퍼" in user_text:
            reply = "무슨 일이 있었나요? 괜찮으세요?"
            emotion = "sadness"
        else:
            reply = f"'{user_text}' 라고 하셨군요."
            emotion = "neutral"
    else:
        if "hello" in user_text.lower():
            reply = "Hello! Nice to meet you."
            emotion = "joy"
        elif "sad" in user_text.lower():
            reply = "What happened? Are you okay?"
            emotion = "sadness"
        else:
            reply = f"You said: '{user_text}'."
            emotion = "neutral"

    # TTS 파일 생성
    filename = f"{uuid.uuid4().hex}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    tts_engine.save_to_file(reply, filepath)
    tts_engine.runAndWait()

    # URL 생성 (static처럼 제공)
    tts_url = url_for("serve_tts", filename=filename, _external=True)

    return jsonify({
        "reply": reply,
        "emotion": emotion,
        "language": lang,
        "tts_url": tts_url
    })

@app.route("/stt", methods=["GET"])
def stt():
    audio = record_audio(duration=5)
    text, lang = transcribe_audio(audio)
    return jsonify({"text": text, "language": lang})

@app.route("/tts/<filename>", methods=["GET"])
def serve_tts(filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype="audio/wav")
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

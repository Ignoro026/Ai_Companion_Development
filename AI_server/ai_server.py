# D:/AI/AICompanion/ai_server/ai_server.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import threading
import time
from pathlib import Path
from flask import Flask, jsonify, send_file, request

from modules.stt_module import WhisperSTT, RealtimeSpeechEngine
from modules.tts_edge import EdgeTTSWrapper


# -----------------------------
# Flask & Paths
# -----------------------------
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

BASE_DIR = Path(__file__).resolve().parent
TTS_DIR = BASE_DIR / "tts_cache"
TTS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Engines (전역 1회 로드)
# -----------------------------
stt = WhisperSTT(model_size="small", device="cpu", compute_type="int8")
rt = None  # /realtime/start 시점에 생성
edge_tts_engine = EdgeTTSWrapper(output_dir=TTS_DIR, voice="ko-KR-SunHiNeural", rate="+0%", pitch="+0%")

pipeline_lock = threading.Lock()

last_result = {
    "user_text": None,
    "reply": None,
    "tts_url": None,
    "language": None
}


# -----------------------------
# 유틸
# -----------------------------
def make_tts_and_url(reply_text: str):
    out_path = edge_tts_engine.synthesize(reply_text)
    return f"/tts_file/{out_path.name}"


# -----------------------------
# 1) 파이프라인
# -----------------------------
@app.route("/pipeline", methods=["GET", "POST"])
def pipeline():
    if not pipeline_lock.acquire(blocking=False):
        return jsonify({"ok": False, "error": "Pipeline busy"}), 409

    try:
        mode = request.args.get("mode", "utterance")  # "utterance" | "timer"
        samplerate = int(request.args.get("samplerate", "16000"))

        if mode == "timer":
            # 고정 시간 녹음(예: 5초)
            print("[Pipeline] Timer mode: recording fixed duration")
            text, lang = stt.record_and_transcribe(duration=5, samplerate=samplerate)
        else:
            # 문장 끝(침묵)까지 기다리되, 최대 10초만 대기
            print("[Pipeline] Utterance mode: waiting for end of speech")
            engine = RealtimeSpeechEngine(
                samplerate=samplerate,
                vad_mode="auto",
                min_utt_sec=1.5,
                end_silence_sec=1.0
            )
            try:
                audio = engine.get_utterance_blocking()
            finally:
                engine.stop()

            if audio is None or len(audio) == 0:
                print("[Pipeline] No speech detected (timeout or silence)")
                return jsonify({"ok": False, "error": "No speech detected"}), 400

            text, lang = stt.transcribe_numpy(audio, samplerate=samplerate)

        if not text:
            print("[Pipeline] Empty STT result")
            return jsonify({"ok": False, "error": "Empty STT result"}), 400

        reply = make_simple_reply(text, lang)
        tts_url = make_tts_and_url(reply)

        last_result.update({
            "user_text": text,
            "reply": reply,
            "tts_url": tts_url,
            "language": lang
        })
        print(f"[Pipeline] User='{text}' | Reply='{reply}'")

        return jsonify({
            "ok": True,
            "language": lang,
            "user_text": text,
            "reply": reply,
            "tts_url": tts_url
        })

    except Exception as e:
        # 에러 잡아서 클라이언트로 내려줌
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

    finally:
        pipeline_lock.release()


# -----------------------------
# 2) 실시간 루프
# -----------------------------
realtime_thread = None
realtime_running = False
realtime_lock = threading.Lock()


def realtime_loop(samplerate=16000):
    global realtime_running, rt
    rt = RealtimeSpeechEngine(samplerate=samplerate, vad_mode="auto", min_utt_sec=1.5, end_silence_sec=1.0)
    try:
        while realtime_running:
            audio = rt.get_utterance_blocking()  # timeout 추가
            if not realtime_running:
                break
            if audio is None or len(audio) == 0:
                continue

            text, lang = stt.transcribe_numpy(audio, samplerate=samplerate)
            if not text:
                continue

            reply = make_simple_reply(text, lang)
            tts_url = make_tts_and_url(reply)

            last_result.update({"user_text": text, "reply": reply, "tts_url": tts_url, "language": lang})
            print(f"[Realtime] User='{text}' | Reply='{reply}'")
    finally:
        if rt:
            rt.stop()
            rt = None


@app.route("/realtime/start", methods=["POST"])
def realtime_start():
    global realtime_thread, realtime_running
    with realtime_lock:
        if realtime_running:
            return jsonify({"ok": True, "msg": "already running"})
        realtime_running = True
        realtime_thread = threading.Thread(target=realtime_loop, kwargs={"samplerate": 16000}, daemon=True)
        realtime_thread.start()
        return jsonify({"ok": True})


@app.route("/realtime/stop", methods=["POST"])
def realtime_stop():
    global realtime_thread, realtime_running
    with realtime_lock:
        if not realtime_running:
            return jsonify({"ok": True, "msg": "not running"})
        realtime_running = False
    if realtime_thread:
        realtime_thread.join(timeout=2.0)
        realtime_thread = None
    return jsonify({"ok": True})


@app.route("/realtime/status", methods=["GET"])
def realtime_status():
    return jsonify({
        "ok": True,
        "is_running": bool(realtime_running),
        "last": last_result
    })


# -----------------------------
# 3) TTS 파일 제공
# -----------------------------
@app.route("/tts_file/<fname>", methods=["GET"])
def tts_file(fname):
    f = TTS_DIR / fname
    if not f.exists():
        return jsonify({"ok": False, "error": "file not found"}), 404
    return send_file(str(f), mimetype="audio/wav")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

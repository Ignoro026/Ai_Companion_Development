# D:/AI/AICompanion/ai_server/ai_server_ws.py
import os
import json
import base64
import asyncio
import threading
from pathlib import Path

import websockets
from websockets.server import serve

from modules.stt_module import WhisperSTT, RealtimeSpeechEngine
try:
    from modules.llm_module import LLMEngine
    _HAVE_LLM = True
except Exception:
    _HAVE_LLM = False

from modules.tts_edge import EdgeTTSWrapper  # voice: ko-KR-SunHiNeural 등

# -----------------------------
# Paths & engines (1회 로드)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
TTS_DIR = BASE_DIR / "tts_cache"
TTS_DIR.mkdir(exist_ok=True)

# STT: faster-whisper (CPU int8 기본)
stt = WhisperSTT(model_size="small", device="cpu", compute_type="int8")

# LLM: 있으면 사용, 없으면 규칙기반
llm = None
if _HAVE_LLM:
    try:
        llm = LLMEngine(model_name="skt/kogpt2-base-v2")  # 원하면 바꿔도 됨
    except Exception as e:
        print(f"[LLM] 로드 실패, 규칙기반으로 대체: {e}")
        llm = None

# TTS: Edge-TTS
tts = EdgeTTSWrapper(output_dir=TTS_DIR, voice="ko-KR-SunHiNeural", rate="+0%", pitch="+0%")

# -----------------------------
# 유틸
# -----------------------------
def simple_rule_reply(user_text: str, lang: str) -> str:
    if lang and lang.startswith("ko"):
        if "안녕" in user_text:
            return "안녕하세요! 만나서 반가워요."
        if "고마" in user_text or "감사" in user_text:
            return "별말씀을요. 도움이 되어서 기뻐요."
        return f"'{user_text}' 라고 하셨군요."
    else:
        ut = user_text.lower()
        if "hello" in ut:
            return "Hello! Nice to meet you."
        if "thanks" in ut or "thank you" in ut:
            return "You're welcome!"
        return f"You said: '{user_text}'."

def synthesize_to_bytes(text: str) -> bytes:
    """
    EdgeTTSWrapper는 파일을 만드니, 만들어진 파일을 읽어서 bytes 반환.
    (WS 전용이라 별도 HTTP 서버 없이 전송하기 위함)
    """
    wav_path = tts.synthesize(text)  # Path 반환
    with open(wav_path, "rb") as f:
        return f.read()

async def send_json(ws, payload: dict):
    await ws.send(json.dumps(payload, ensure_ascii=False))

# -----------------------------
# 세션별 음성 루프 (스레드)
# -----------------------------
class VoiceSession:
    def __init__(self, ws, samplerate=16000):
        self.ws = ws
        self.samplerate = samplerate
        self.rt_engine = None
        self.alive = True
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.alive = False
        if self.rt_engine:
            try:
                self.rt_engine.stop()
            except Exception:
                pass
            self.rt_engine = None
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

    def _loop(self):
        # 주: websockets는 asyncio 전용이므로, 스레드에서 직접 ws.send 불가
        # -> thread-safe하게 asyncio로 보낼 수 있도록 loop 참조를 잡음
        loop = asyncio.get_event_loop()

        def send_safe(payload):
            asyncio.run_coroutine_threadsafe(send_json(self.ws, payload), loop)

        try:
            self.rt_engine = RealtimeSpeechEngine(
                samplerate=self.samplerate,
                vad_mode="auto",
                min_utt_sec=1.5,
                end_silence_sec=1.0
            )

            # 최초 상태 통지
            send_safe({"type": "status", "ok": True, "msg": "listening"})

            while self.alive:
                audio = self.rt_engine.get_utterance_blocking()
                if not self.alive:
                    break
                if audio is None or len(audio) == 0:
                    continue

                # STT
                try:
                    text, lang = stt.transcribe_numpy(audio, samplerate=self.samplerate)
                except Exception as e:
                    send_safe({"type": "error", "stage": "stt", "message": str(e)})
                    continue

                if not text:
                    # 무음 또는 너무 짧음
                    send_safe({"type": "stt", "ok": False, "msg": "empty"})
                    continue

                send_safe({"type": "stt", "ok": True, "language": lang, "text": text})

                # LLM
                try:
                    if llm is not None:
                        reply = llm.generate_reply(text)
                    else:
                        reply = simple_rule_reply(text, lang)
                except Exception as e:
                    send_safe({"type": "error", "stage": "llm", "message": str(e)})
                    continue

                send_safe({"type": "llm", "ok": True, "reply": reply})

                # TTS (bytes -> base64)
                try:
                    wav_bytes = synthesize_to_bytes(reply)
                    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
                    send_safe({
                        "type": "tts",
                        "ok": True,
                        "mime": "audio/wav",
                        "audio_b64": audio_b64,
                        "reply": reply,
                        "language": lang
                    })
                except Exception as e:
                    send_safe({"type": "error", "stage": "tts", "message": str(e)})
                    continue

        finally:
            send_safe({"type": "status", "ok": True, "msg": "stopped"})

# -----------------------------
# WebSocket 핸들러
# -----------------------------
async def ws_handler(ws):
    """
    - 클라이언트가 연결되면 서버 마이크로 실시간 청취 시작
    - 받은 문장마다 STT/LLM/TTS 결과를 순서대로 push
    - 클라이언트가 'stop' 메시지 보내면 종료
    """
    session = VoiceSession(ws, samplerate=16000)
    session.start()
    try:
        async for msg in ws:
            # 클라이언트 제어 메시지 (JSON 권장)
            try:
                data = json.loads(msg)
            except Exception:
                data = {"type": "raw", "value": msg}

            if isinstance(data, dict) and data.get("cmd") == "stop":
                await send_json(ws, {"type": "status", "ok": True, "msg": "stopping"})
                break

            # samplerate 변경 등 옵션
            if data.get("cmd") == "set" and "samplerate" in data:
                # 간단 구현: 다음 연결부터 반영 권장
                await send_json(ws, {"type": "warn", "msg": "samplerate change requires reconnect"})
    finally:
        session.stop()

# -----------------------------
# 메인
# -----------------------------
async def main():
    host = "127.0.0.1"
    port = 5001
    print(f"[WS] Listening on ws://{host}:{port}/ws")
    async with serve(ws_handler, host, port, ping_interval=20, ping_timeout=20, path="/ws"):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

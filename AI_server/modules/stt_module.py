import queue
import time
import wave
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd

try:
    import webrtcvad
    _HAVE_VAD = True
except Exception:
    _HAVE_VAD = False

from faster_whisper import WhisperModel


class WhisperSTT:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def _transcribe_wav_path(self, wav_path: str):
        segments, info = self.model.transcribe(wav_path, language=None)
        text = " ".join([seg.text for seg in segments]).strip()
        return text, (info.language or "auto")

    def transcribe_numpy(self, audio_int16: np.ndarray, samplerate=16000):
        """
        numpy int16 PCM → Whisper 변환
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            with wave.open(tmpfile.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16
                wf.setframerate(samplerate)
                wf.writeframes(audio_int16.tobytes())
            tmp_wav = tmpfile.name
        return self._transcribe_wav_path(tmp_wav)

    def record_and_transcribe(self, duration=5, samplerate=16000):
        """
        Blocking 녹음 → 변환 (테스트용)
        """
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
        sd.wait()
        audio = np.squeeze(audio)
        # ✅ 오타 수정: samplerate로 넘김
        return self.transcribe_numpy(audio, samplerate=samplerate)


# --- 실시간 엔진: webrtcvad(있으면) 또는 RMS 침묵 감지 ---
class RealtimeSpeechEngine:
    def __init__(self, samplerate=16000, vad_mode="auto", min_utt_sec=1.5, end_silence_sec=1.0):
        self.samplerate = samplerate
        self.min_utt_sec = float(min_utt_sec)
        self.end_silence_sec = float(end_silence_sec)
        self.q = queue.Queue()
        self.stream = None
        self.running = False

        # --- VAD 선택 ---
        self.use_vad = False
        if vad_mode == "auto" and _HAVE_VAD:
            self.use_vad = True
        elif vad_mode == "webrtcvad":
            self.use_vad = True

        if self.use_vad:
            self.vad = webrtcvad.Vad(2)  # 0~3 (3이 가장 민감)
            self.frame_ms = 20           # 10/20/30ms만 허용
            self.frame_len = int(self.samplerate * (self.frame_ms/1000.0))
        else:
            # RMS 기반
            self.frame_len = int(self.samplerate * 0.02)  # 20ms
            self.rms_threshold = 0.01

        self._start_stream()

    # 입력 콜백
    def _callback(self, indata, frames, time_, status):
        if status:
            print(status)
        # float32 → int16 변환
        data = (indata[:, 0] * 32767.0).astype(np.int16)
        self.q.put(data)

    def _start_stream(self):
        if self.stream:
            return
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            callback=self._callback
        )
        self.stream.start()
        self.running = True

    def stop(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
            except Exception:
                pass
            self.stream.close()
            self.stream = None

    def _vad_is_speech(self, frame_int16: np.ndarray) -> bool:
        return self.vad.is_speech(frame_int16.tobytes(), self.samplerate)

    def _rms_is_speech(self, frame_int16: np.ndarray) -> bool:
        f = frame_int16.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(f*f) + 1e-9)
        return rms > self.rms_threshold

    def get_utterance_blocking(self):
        """
        완성된 한 문장(utterance) 단위 음성 데이터를 반환.
        문장이 끝나기 전까지는 blocking 상태.
        """
        if not self.running:
            time.sleep(0.05)
            return None

        collected = []
        voiced_started = False
        voiced_start_time = None
        last_voice_time = None

        while self.running:
            try:
                frame = self.q.get(timeout=0.1)
            except queue.Empty:
                frame = None

            if frame is None:
                # 침묵 시간 체크
                if voiced_started and last_voice_time and (time.time() - last_voice_time) >= self.end_silence_sec:
                    audio = np.concatenate(collected) if collected else np.array([], dtype=np.int16)
                    min_len = int(self.samplerate * self.min_utt_sec)
                    if len(audio) >= min_len:
                        return audio
                    else:
                        # 너무 짧으면 폐기
                        collected, voiced_started = [], False
                        voiced_start_time, last_voice_time = None, None
                continue

            # frame을 잘라 VAD 적용
            for i in range(0, len(frame), self.frame_len):
                sub = frame[i:i+self.frame_len]
                if len(sub) < self.frame_len:
                    continue

                if self.use_vad:
                    speech = self._vad_is_speech(sub)
                else:
                    speech = self._rms_is_speech(sub)

                if speech:
                    if not voiced_started:
                        voiced_started = True
                        voiced_start_time = time.time()
                    collected.append(sub)
                    last_voice_time = time.time()
                else:
                    collected.append(sub)
                    # 발화 종료는 상단 timeout에서 판정

        return None

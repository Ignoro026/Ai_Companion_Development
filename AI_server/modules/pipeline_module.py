from .stt_module import WhisperSTT, RealtimeSpeechEngine
from .llm_module import LLMEngine
from .tts_module import EdgeTTS
from .memory_module import Memory

class Pipeline:
    def __init__(self):
        self.stt = WhisperSTT(model_size="small", device="cpu")
        self.engine = RealtimeSpeechEngine()
        self.llm = LLMEngine(model_name="gpt2")
        self.tts = EdgeTTS()
        self.memory = Memory(max_turns=5)

    def run_once(self):
        # 1. 발화 감지 + 텍스트 변환
        audio = self.engine.get_utterance_blocking()
        if audio is None:
            return None

        text, lang = self.stt.transcribe_numpy(audio)
        if not text:
            return None

        # 2. LLM 응답 생성
        history = self.memory.get_context()
        reply = self.llm.generate_reply(text, history)

        # 3. 메모리 업데이트
        self.memory.append(user=text, bot=reply)

        # 4. TTS 변환
        wav_path = self.tts.text_to_speech(reply)

        return {"user_text": text, "bot_text": reply, "audio_path": wav_path}

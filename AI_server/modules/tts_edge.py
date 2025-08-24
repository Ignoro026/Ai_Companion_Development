# D:/AI/AICompanion/ai_server/modules/tts_edge.py
import uuid
import asyncio
from pathlib import Path
import edge_tts


class EdgeTTSWrapper:
    """
    Edge TTS 간단 래퍼.
    - 기본 한국어 여성: ko-KR-SunHiNeural (자연스러움·명료함)
    - WAV 출력(riff-24khz-16bit-mono-pcm)
    """
    def __init__(
        self,
        output_dir: Path,
        voice: str = "ko-KR-SunHiNeural",
        rate: str = "+0%",
        pitch: str = "+0%",
        audio_format: str = "riff-24khz-16bit-mono-pcm",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.audio_format = audio_format

    async def _synthesize_async(self, text: str, out_path: Path):
        tts = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
        )
        # audio_format은 save()에 넘겨야 적용됨
        await tts.save(str(out_path), audio_format=self.audio_format)

    def synthesize(self, text: str) -> Path:
        out_path = self.output_dir / f"{uuid.uuid4().hex}.wav"
        # Flask 스레드 내에서 간단히 실행
        asyncio.run(self._synthesize_async(text, out_path))
        return out_path

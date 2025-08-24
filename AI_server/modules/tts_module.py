import uuid
import asyncio
from pathlib import Path
import edge_tts

# Windows에서 Flask와 asyncio 충돌 회피가 필요하면:
# import sys
# if sys.platform.startswith("win"):
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def _synthesize_async(text: str, out_path: Path, voice: str):
    tts = edge_tts.Communicate(text, voice)
    await tts.save(str(out_path))

def synthesize_tts(text: str, out_dir: Path, voice="ko-KR-SunHiNeural") -> Path:
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"{uuid.uuid4().hex}.wav"
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_synthesize_async(text, out_path, voice))
    finally:
        loop.close()
    return out_path

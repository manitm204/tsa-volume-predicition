#!/usr/bin/env python3
"""Play all liked voices for comparison. Run: python scripts/voice_audition.py"""
import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

SAMPLE = (
    "Our weekly prediction has moved to 2.68 million, keeping us well aligned "
    "with Kalshi's implied level. "
    "We're sitting on two active weekly bets with a meaningful 22 percent edge "
    "on our main position at the 2.65 million threshold."
)

EDGE_VOICES = [
#    ("en-GB-RyanNeural",                "Ryan — British Male",                  1),
#    ("en-US-ChristopherNeural",         "Christopher — American Male",          2),
#    ("en-AU-WilliamMultilingualNeural", "William — Australian Male  ★ FAVORITE", 3),
]

ELEVEN_VOICES = [
    ("Daniel", "Daniel — British Male (Steady Broadcaster)",      4),
    ("Brian",  "Brian — American Male (Deep, Resonant)",          5),
]

TABLE = """\
┌────┬──────────────────────────────────────────┬──────────────────────┬──────────────────────────────┐
│ #  │ Voice                                    │ Source               │ Pricing                      │
├────┼──────────────────────────────────────────┼──────────────────────┼──────────────────────────────┤
│  1 │ Ryan          (en-GB-RyanNeural)          │ Edge TTS (Microsoft) │ Free — no API key            │
│  2 │ Christopher   (en-US-ChristopherNeural)   │ Edge TTS (Microsoft) │ Free — no API key            │
│  3 │ William ★     (en-AU-WilliamMultilingual) │ Edge TTS (Microsoft) │ Free — no API key            │
│  4 │ Daniel        Steady Broadcaster          │ ElevenLabs           │ Free tier: 10k chars/month   │
│  5 │ Brian         Deep, Resonant & Comforting │ ElevenLabs           │ Free tier: 10k chars/month   │
└────┴──────────────────────────────────────────┴──────────────────────┴──────────────────────────────┘"""


def _play_mp3(path: str) -> None:
    subprocess.run(
        ["gst-play-1.0", "--no-interactive", path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def _play_edge(voice_id: str, label: str, num: int) -> None:
    import edge_tts
    print(f"\n[{num}/5]  {label}")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        path = f.name
    try:
        await edge_tts.Communicate(SAMPLE, voice_id).save(path)
        _play_mp3(path)
    finally:
        Path(path).unlink(missing_ok=True)


def _play_elevenlabs(voice_name: str, label: str, num: int) -> None:
    key = os.getenv("ELEVENLABS_API_KEY", "")
    if not key:
        print(f"\n[{num}/5]  {label}  ← SKIPPED (export ELEVENLABS_API_KEY=... to hear this)")
        return
    print(f"\n[{num}/5]  {label}")
    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=key)
        # include legacy/pre-made voices
        all_voices = client.voices.get_all(show_legacy=True).voices
        match = next(
            (v for v in all_voices if v.name.split(" - ")[0].strip().lower() == voice_name.lower()),
            None,
        )
        if not match:
            names = [v.name for v in all_voices]
            print(f"  Voice '{voice_name}' not found. Available: {', '.join(names)}")
            return
        audio_iter = client.text_to_speech.convert(
            voice_id=match.voice_id,
            text=SAMPLE,
            model_id="eleven_turbo_v2_5",
        )
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            for chunk in audio_iter:
                f.write(chunk)
            path = f.name
        try:
            _play_mp3(path)
        finally:
            Path(path).unlink(missing_ok=True)
    except Exception as e:
        print(f"  Error: {e}")


async def main() -> None:
    print(TABLE)
    print(f'\nSample: "{SAMPLE}"\n')
    input("Press Enter to begin the audition...")

    for voice_id, label, num in EDGE_VOICES:
        await _play_edge(voice_id, label, num)
        await asyncio.sleep(0.3)

    for voice_name, label, num in ELEVEN_VOICES:
        _play_elevenlabs(voice_name, label, num)
        await asyncio.sleep(0.3)

    print("\nAudition complete.")


if __name__ == "__main__":
    asyncio.run(main())

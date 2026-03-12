"""
One-time script: generate greeting audio from ElevenLabs and save as greeting.pcm
Run locally: python generate_greeting.py
Commit greeting.pcm to the repo so Render can load it without calling ElevenLabs HTTP.
"""
import asyncio
import os
import aiohttp
from dotenv import load_dotenv

load_dotenv(override=True)

GREETING = "வணக்கம்! நான் உங்கள் சேவை முகவர். நான் எப்படி உதவலாம்?"
OUTPUT_FILE = "greeting.pcm"


async def main():
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not voice_id or not api_key:
        print("ERROR: ELEVENLABS_VOICE_ID or ELEVENLABS_API_KEY not set")
        return

    print(f"Generating greeting audio for: {GREETING!r}")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=pcm_8000",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json={
                "text": GREETING,
                "model_id": "eleven_flash_v2_5",
                "language_code": "ta",
            },
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status == 200:
                pcm_bytes = await resp.read()
                with open(OUTPUT_FILE, "wb") as f:
                    f.write(pcm_bytes)
                print(f"Saved {len(pcm_bytes)} bytes PCM to {OUTPUT_FILE}")
            else:
                text = await resp.text()
                print(f"ERROR HTTP {resp.status}: {text[:300]}")


asyncio.run(main())

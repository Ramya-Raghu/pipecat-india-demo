"""
Simulates a Plivo WebSocket connection to test the bot pipeline on Render.
Sends a fake Plivo start event and silent audio, then listens for bot audio back.
"""
import asyncio
import base64
import json
import time
import struct

import websockets

WS_URL = "wss://pipecat-india-demo.onrender.com/ws"
FAKE_CALL_ID = "test-call-00000000"
FAKE_STREAM_ID = "test-stream-00000000"

# 160 bytes of silence = 20ms of 8kHz μ-law audio (Plivo sends 20ms chunks)
SILENCE_FRAME = base64.b64encode(bytes([0xFF] * 160)).decode()


def plivo_start_event():
    return json.dumps({
        "event": "start",
        "start": {
            "streamId": FAKE_STREAM_ID,
            "callId": FAKE_CALL_ID,
            "customParameters": {},
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000},
        }
    })


def plivo_media_event(seq: int):
    return json.dumps({
        "event": "media",
        "sequenceNumber": str(seq),
        "media": {
            "chunk": str(seq),
            "timestamp": str(seq * 20),
            "payload": SILENCE_FRAME,
        }
    })


async def run_test():
    body = base64.b64encode(json.dumps({"mode": "inbound"}).encode()).decode()
    url = f"{WS_URL}?body={body}"

    print(f"Connecting to {url}")
    start = time.time()

    async with websockets.connect(url) as ws:
        print(f"[{time.time()-start:.2f}s] WebSocket connected")

        # Send Plivo start event
        await ws.send(plivo_start_event())
        print(f"[{time.time()-start:.2f}s] Sent start event")

        # Send silence audio for up to 30 seconds, listen for bot audio
        audio_received = False
        seq = 1

        async def send_silence():
            nonlocal seq
            while True:
                await ws.send(plivo_media_event(seq))
                seq += 1
                await asyncio.sleep(0.02)  # 20ms chunks

        async def receive_messages():
            nonlocal audio_received
            async for msg in ws:
                data = json.loads(msg)
                event = data.get("event")
                # Plivo outbound audio uses "playAudio"; inbound echo uses "media"
                if event in ("media", "playAudio"):
                    if not audio_received:
                        print(f"[{time.time()-start:.2f}s] ✅ Bot sent first audio frame! (event={event})")
                        audio_received = True
                elif event == "mark":
                    print(f"[{time.time()-start:.2f}s] Mark: {data.get('mark', {}).get('name')}")
                elif event:
                    print(f"[{time.time()-start:.2f}s] Event: {event}")

        try:
            await asyncio.wait_for(
                asyncio.gather(send_silence(), receive_messages()),
                timeout=20
            )
        except asyncio.TimeoutError:
            if audio_received:
                print(f"\n✅ Bot responded with audio. Test passed.")
            else:
                print(f"\n❌ No audio from bot after 20 seconds.")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[{time.time()-start:.2f}s] Connection closed: {e}")
            if audio_received:
                print("✅ Bot responded before disconnect.")
            else:
                print("❌ Connection closed before bot responded.")

if __name__ == "__main__":
    asyncio.run(run_test())

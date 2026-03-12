"""
Combined inbound + outbound server on port 7860.

Inbound:  GET  /inbound  → Plivo answer webhook (set as your number's Answer URL)
Outbound: POST /start    → trigger an outbound call
          GET  /answer   → Plivo answer webhook for outbound calls
          WS   /ws       → shared WebSocket handler for both
"""

import base64
import json
import os
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import Response

load_dotenv(override=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.session = aiohttp.ClientSession()
    yield
    await app.state.session.close()


app = FastAPI(title="Pipecat India Demo", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    required = [
        "NVIDIA_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY",
        "ELEVENLABS_VOICE_ID", "PLIVO_AUTH_ID", "PLIVO_AUTH_TOKEN", "PLIVO_PHONE_NUMBER",
    ]
    missing = [k for k in required if not os.getenv(k)]

    # Test NVIDIA gRPC reachability
    nvidia_ok = False
    nvidia_error = None
    try:
        import grpc
        channel = grpc.secure_channel(
            "grpc.nvcf.nvidia.com:443",
            grpc.ssl_channel_credentials(),
        )
        grpc.channel_ready_future(channel).result(timeout=5)
        nvidia_ok = True
        channel.close()
    except Exception as e:
        nvidia_error = str(e)

    return JSONResponse({
        "status": "ok" if not missing else "missing_env_vars",
        "missing": missing,
        "public_url": os.getenv("PUBLIC_URL", "(not set)"),
        "nvidia_grpc": "ok" if nvidia_ok else f"error: {nvidia_error}",
    })


def build_stream_xml(ws_url: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">
    {ws_url}
  </Stream>
</Response>"""


# ─── INBOUND ────────────────────────────────────────────────────────────────

@app.get("/inbound")
async def inbound_webhook(
    request: Request,
    CallUUID: str = Query(None),
    From: str = Query(None),
    To: str = Query(None),
):
    """Plivo calls this when someone dials your number."""
    print(f"[INBOUND] Call from {From} → {To}, UUID={CallUUID}")
    host = request.headers.get("host")
    body = base64.b64encode(json.dumps({"mode": "inbound", "from": From, "to": To}).encode()).decode()
    ws_url = f"wss://{host}/ws?body={body}"
    return Response(content=build_stream_xml(ws_url), media_type="application/xml")


# ─── OUTBOUND ───────────────────────────────────────────────────────────────

@app.post("/start")
async def start_outbound_call(request: Request) -> JSONResponse:
    """Trigger an outbound call. Body: {"phone_number": "+91XXXXXXXXXX", "public_url": "https://xxxx.ngrok-free.app"}"""
    data = await request.json()
    phone_number = data.get("phone_number")
    if not phone_number:
        raise HTTPException(status_code=400, detail="Missing phone_number")

    # Use explicit public_url if provided, otherwise fall back to host header
    public_url = data.get("public_url") or os.getenv("PUBLIC_URL")
    if public_url:
        answer_url = f"{public_url.rstrip('/')}/answer"
    else:
        host = request.headers.get("host")
        protocol = "http" if host.startswith(("localhost", "127.")) else "https"
        answer_url = f"{protocol}://{host}/answer"

    auth_id = os.getenv("PLIVO_AUTH_ID")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN")
    auth = aiohttp.BasicAuth(auth_id, auth_token)

    async with request.app.state.session.post(
        f"https://api.plivo.com/v1/Account/{auth_id}/Call/",
        json={
            "to": phone_number,
            "from": os.getenv("PLIVO_PHONE_NUMBER"),
            "answer_url": answer_url,
            "answer_method": "GET",
        },
        auth=auth,
    ) as resp:
        if resp.status != 201:
            raise HTTPException(status_code=500, detail=await resp.text())
        result = await resp.json()

    print(f"[OUTBOUND] Call queued to {phone_number}, result={result}")
    return JSONResponse({"status": "call_initiated", "phone_number": phone_number, "result": result})


@app.get("/answer")
async def outbound_answer(request: Request, CallUUID: str = Query(None)) -> Response:
    """Plivo hits this once the outbound call is answered."""
    print(f"[OUTBOUND] Call answered, UUID={CallUUID}")
    host = request.headers.get("host")
    body = base64.b64encode(json.dumps({"mode": "outbound"}).encode()).decode()
    ws_url = f"wss://{host}/ws?body={body}"
    return Response(content=build_stream_xml(ws_url), media_type="application/xml")


# ─── SHARED WEBSOCKET ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, body: str = Query(None)):
    await websocket.accept()

    mode = "inbound"
    if body:
        try:
            decoded = json.loads(base64.b64decode(body).decode())
            mode = decoded.get("mode", "inbound")
        except Exception:
            pass

    print(f"[WS] Connection accepted, mode={mode}")

    try:
        from bot import bot
        from pipecat.runner.types import WebSocketRunnerArguments

        runner_args = WebSocketRunnerArguments(websocket=websocket)
        await bot(runner_args, mode=mode)
    except Exception as e:
        print(f"[WS] Error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.close()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

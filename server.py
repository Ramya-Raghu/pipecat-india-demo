"""
Combined inbound + outbound server on port 7860.

Inbound:  GET  /inbound  → Plivo answer webhook (set as your number's Answer URL)
Outbound: POST /start    → trigger an outbound call
          GET  /answer   → Plivo answer webhook for outbound calls
          WS   /ws       → shared WebSocket handler for both
"""

import asyncio
import base64
import json
import os
import time
from collections import deque
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
    # Pre-generate greeting audio at startup so the first call has it ready.
    from bot import _ensure_greeting_audio
    asyncio.create_task(_ensure_greeting_audio())
    yield
    await app.state.session.close()


app = FastAPI(title="Pipecat India Demo", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

call_logs: deque = deque(maxlen=5)  # store last 5 call log traces


@app.get("/last-call")
async def last_call() -> JSONResponse:
    return JSONResponse({"calls": list(call_logs)})


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
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        '<Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">'
        f"{ws_url}"
        "</Stream>"
        "</Response>"
    )


# ─── INBOUND ────────────────────────────────────────────────────────────────

@app.get("/inbound")
async def inbound_webhook(
    request: Request,
    CallUUID: str = Query(None),
    From: str = Query(None),
    To: str = Query(None),
):
    """Plivo calls this when someone dials your number."""
    host = request.headers.get("host")
    call_logs.append({"mode": "inbound-webhook", "from": From, "to": To, "uuid": CallUUID, "host": host, "t0": time.time(), "events": []})
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

    log = {"mode": mode, "events": [], "t0": time.time()}
    call_logs.append(log)

    def ev(msg):
        log["events"].append({"t": round(time.time() - log["t0"], 3), "msg": msg})
        print(f"[WS] {msg}")

    ev("connection accepted")

    try:
        from bot import bot
        from pipecat.runner.types import WebSocketRunnerArguments

        ev("calling parse_telephony_websocket")
        runner_args = WebSocketRunnerArguments(websocket=websocket)
        ev("starting bot")
        await bot(runner_args, mode=mode)
        ev("bot finished")
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        ev(f"ERROR: {e} | {err[-300:]}")
        await websocket.close()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

"""
Outbound server: POST /start with a phone number → Plivo dials out → bot pipeline.
Usage: curl -X POST http://localhost:7861/start -H "Content-Type: application/json" \
            -d '{"phone_number": "+919XXXXXXXXX"}'
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


app = FastAPI(title="Plivo Outbound Server", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def make_plivo_call(session, to_number: str, from_number: str, answer_url: str):
    auth_id = os.getenv("PLIVO_AUTH_ID")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN")
    url = f"https://api.plivo.com/v1/Account/{auth_id}/Call/"
    auth = aiohttp.BasicAuth(auth_id, auth_token)
    async with session.post(
        url,
        json={"to": to_number, "from": from_number, "answer_url": answer_url, "answer_method": "GET"},
        auth=auth,
    ) as resp:
        if resp.status != 201:
            raise Exception(f"Plivo error ({resp.status}): {await resp.text()}")
        return await resp.json()


@app.post("/start")
async def start_outbound_call(request: Request) -> JSONResponse:
    data = await request.json()
    phone_number = data.get("phone_number")
    if not phone_number:
        raise HTTPException(status_code=400, detail="Missing phone_number")

    host = request.headers.get("host")
    protocol = "http" if host.startswith("localhost") or host.startswith("127.") else "https"
    answer_url = f"{protocol}://{host}/answer"

    result = await make_plivo_call(
        session=request.app.state.session,
        to_number=phone_number,
        from_number=os.getenv("PLIVO_PHONE_NUMBER"),
        answer_url=answer_url,
    )
    return JSONResponse({"status": "call_initiated", "phone_number": phone_number, "result": result})


@app.get("/answer")
async def answer_xml(request: Request) -> Response:
    host = request.headers.get("host")
    ws_url = f"wss://{host}/ws"
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">
    {ws_url}
  </Stream>
</Response>"""
    return Response(content=xml, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, body: str = Query(None)):
    await websocket.accept()
    try:
        from bot import bot
        from pipecat.runner.types import WebSocketRunnerArguments

        runner_args = WebSocketRunnerArguments(websocket=websocket)
        await bot(runner_args, mode="outbound")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7861)

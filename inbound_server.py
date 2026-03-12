"""
Inbound server: Plivo calls your number → webhook → WebSocket → bot pipeline.
Configure your Plivo number's Answer URL to: http://<ngrok-url>/
"""

import base64
import json
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request, WebSocket
from starlette.responses import Response

load_dotenv(override=True)

app = FastAPI(title="Plivo Inbound Server")


@app.get("/")
async def inbound_webhook(
    request: Request,
    CallUUID: str = Query(None),
    From: str = Query(None),
    To: str = Query(None),
):
    host = request.headers.get("host")
    body_data = {}
    if From:
        body_data["from"] = From
    if To:
        body_data["to"] = To

    body_json = json.dumps(body_data)
    body_encoded = base64.b64encode(body_json.encode()).decode()
    ws_url = f"wss://{host}/ws?body={body_encoded}"

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

    body_data = {}
    if body:
        try:
            body_data = json.loads(base64.b64decode(body).decode())
        except Exception:
            pass

    try:
        from bot import bot
        from pipecat.runner.types import WebSocketRunnerArguments

        runner_args = WebSocketRunnerArguments(websocket=websocket)
        await bot(runner_args, mode="inbound")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

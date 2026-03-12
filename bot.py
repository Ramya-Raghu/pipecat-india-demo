import asyncio
import audioop
import base64
import json
import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.utils.time import time_now_iso8601
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_turn_strategies import (
    LocalSmartTurnAnalyzerV3,
    TurnAnalyzerUserTurnStopStrategy,
    UserTurnStrategies,
    VADUserTurnStartStrategy,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

# NVIDIA NIM function ID for parakeet-rnnt-1.1b-multilingual (supports hi-IN)
NVIDIA_MULTILINGUAL_FUNCTION_ID = "71203149-d3b7-4460-8231-1be2543a1fca"

# ─── GREETING PRE-GENERATION ─────────────────────────────────────────────────
# Key insight: send the greeting DIRECTLY via WebSocket BEFORE the pipeline starts.
# This avoids waiting for ElevenLabs WebSocket cold-start (0.5-1s on Render).
#
# Primary strategy: load pre-generated PCM from greeting.pcm (committed to repo).
# This file is generated locally with generate_greeting.py and committed so that
# Render can load it instantly at startup — no HTTP call needed.
#
# Fallback: if greeting.pcm is missing, call ElevenLabs HTTP API at startup.
# NOTE: ElevenLabs HTTP API is blocked on Render free tier (HTTP 401), so the
# committed file approach is required for sub-2s greeting on Render.

_GREETING = "வணக்கம்! நான் உங்கள் சேவை முகவர். நான் எப்படி உதவலாம்?"
_greeting_pcm: bytes | None = None
_greeting_error: str | None = None

# Load from committed file immediately at import time (synchronous, zero latency).
_GREETING_PCM_FILE = os.path.join(os.path.dirname(__file__), "greeting.pcm")
if os.path.exists(_GREETING_PCM_FILE):
    with open(_GREETING_PCM_FILE, "rb") as _f:
        _greeting_pcm = _f.read()
    logger.info(f"Loaded greeting PCM from file: {len(_greeting_pcm)} bytes")


async def _ensure_greeting_audio() -> None:
    """Fallback: pre-generate greeting via ElevenLabs HTTP API if greeting.pcm is missing."""
    global _greeting_pcm, _greeting_error
    if _greeting_pcm is not None:
        return
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not voice_id or not api_key:
        _greeting_error = "Missing ELEVENLABS_VOICE_ID or ELEVENLABS_API_KEY"
        logger.warning(f"Cannot pre-generate greeting: {_greeting_error}")
        return
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=pcm_8000",
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "text": _GREETING,
                    "model_id": "eleven_flash_v2_5",
                    "language_code": "ta",
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    _greeting_pcm = await resp.read()
                    _greeting_error = None
                    logger.info(f"Pre-generated greeting audio: {len(_greeting_pcm)} bytes PCM")
                else:
                    text = await resp.text()
                    _greeting_error = f"HTTP {resp.status}: {text[:300]}"
                    logger.warning(f"ElevenLabs pre-gen failed: {_greeting_error}")
    except Exception as e:
        _greeting_error = str(e)
        logger.warning(f"Failed to pre-generate greeting audio: {e}")


async def _send_greeting_via_websocket(websocket, stream_id: str) -> None:
    """Send pre-generated greeting PCM directly over the Plivo WebSocket.

    Bypasses the pipeline entirely — runs BEFORE NVIDIA STT or ElevenLabs WS
    are initialized. PCM 16-bit 8kHz → μ-law → base64 → Plivo playAudio JSON.
    """
    if not _greeting_pcm:
        return
    try:
        # audioop.lin2ulaw: PCM 16-bit → μ-law 8-bit (same sample rate, no resampling)
        ulaw_data = audioop.lin2ulaw(_greeting_pcm, 2)
        payload = base64.b64encode(ulaw_data).decode("utf-8")
        msg = json.dumps({
            "event": "playAudio",
            "media": {
                "contentType": "audio/x-mulaw",
                "sampleRate": 8000,
                "payload": payload,
            },
            "streamId": stream_id,
        })
        await websocket.send_text(msg)
        logger.info(f"Greeting sent directly via WebSocket ({len(ulaw_data)} bytes μ-law)")
    except Exception as e:
        logger.warning(f"Failed to send greeting via WebSocket: {e}")


# ─── NVIDIA STT PATCH ────────────────────────────────────────────────────────


class FinalizingNvidiaSTTService(NvidiaSTTService):
    """NvidiaSTTService that marks final transcriptions as finalized=True.

    Pipecat's TurnAnalyzerUserTurnStopStrategy requires finalized=True to trigger
    immediately after the smart turn model signals COMPLETE. The upstream
    NvidiaSTTService never sets this flag, so we add it here.
    """

    async def _handle_response(self, response):
        for result in response.results:
            if result and not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript
            if transcript and len(transcript) > 0:
                if result.is_final:
                    await self.stop_processing_metrics()
                    await self.push_frame(
                        TranscriptionFrame(
                            transcript,
                            self._user_id,
                            time_now_iso8601(),
                            self._settings.language,
                            result=result,
                            finalized=True,
                        )
                    )
                    await self._handle_transcription(
                        transcript=transcript,
                        is_final=result.is_final,
                        language=self._settings.language,
                    )
                else:
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            transcript,
                            self._user_id,
                            time_now_iso8601(),
                            self._settings.language,
                            result=result,
                        )
                    )


async def run_bot(
    transport: BaseTransport,
    handle_sigint: bool,
    mode: str = "inbound",
    greeting_already_sent: bool = False,
):
    stt = FinalizingNvidiaSTTService(
        api_key=os.getenv("NVIDIA_API_KEY"),
        model_function_map={
            "function_id": NVIDIA_MULTILINGUAL_FUNCTION_ID,
            "model_name": "parakeet-rnnt-1.1b-multilingual",
        },
        params=NvidiaSTTService.InputParams(language=Language.TA_IN),
    )
    # Select the Indic model type to enable Tamil (ta-IN) support
    stt._custom_configuration = "type:indic"

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        model="eleven_flash_v2_5",
        language="ta",  # Tamil — improves phoneme quality with multilingual model
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful Tamil-speaking customer service agent. "
                "Respond in Tamil by default. Keep responses short — one or two sentences. "
                "Do not use special characters or formatting as responses will be spoken aloud. "
                "If the user speaks in English, respond in English."
            ),
        },
        # Greeting is pre-sent via WebSocket before pipeline starts.
        # Add it to context so follow-up turns have conversational history.
        {"role": "assistant", "content": _GREETING},
    ]

    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=0.1, start_secs=0.1)
            ),
            # VAD-only start prevents late NVIDIA STT frames from false-interrupting.
            # Smart turn stop ends the turn as soon as the model says COMPLETE.
            # Fallback timeout reduced to 1.5s (from default 5s).
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[TurnAnalyzerUserTurnStopStrategy(
                    turn_analyzer=LocalSmartTurnAnalyzerV3()
                )],
            ),
            user_turn_stop_timeout=1.5,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Call connected ({mode})")
        if not greeting_already_sent:
            # PCM wasn't ready pre-pipeline — use TTS WebSocket as fallback.
            # By now (3-4s into pipeline startup), ElevenLabs WS is established.
            logger.warning("Pre-pipeline greeting skipped, sending via TTSSpeakFrame fallback")
            await task.queue_frames([TTSSpeakFrame(text=_GREETING)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Call disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments, mode: str = "inbound"):
    # Kick off greeting pre-generation early (fire-and-forget, cached after first call).
    if _greeting_pcm is None:
        asyncio.create_task(_ensure_greeting_audio())

    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Transport detected: {transport_type}, mode: {mode}")

    serializer = PlivoFrameSerializer(
        stream_id=call_data["stream_id"],
        call_id=call_data["call_id"],
        auth_id=os.getenv("PLIVO_AUTH_ID", ""),
        auth_token=os.getenv("PLIVO_AUTH_TOKEN", ""),
    )

    # Send greeting BEFORE pipeline starts — avoids 3-4s NVIDIA STT gRPC wait.
    # If PCM isn't ready yet (first call after server start), skip here and fall
    # back to TTSSpeakFrame in on_client_connected (still works, just ~3-4s slower).
    greeting_sent = False
    if _greeting_pcm:
        await _send_greeting_via_websocket(runner_args.websocket, call_data["stream_id"])
        greeting_sent = True
    else:
        logger.warning("Greeting PCM not ready — will use TTSSpeakFrame fallback via pipeline")

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    await run_bot(transport, runner_args.handle_sigint, mode=mode, greeting_already_sent=greeting_sent)

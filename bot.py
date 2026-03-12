import asyncio
import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMRunFrame,
    OutputAudioRawFrame,
    StartFrame,
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
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
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
# Pre-generate the greeting via ElevenLabs HTTP API at server startup.
# This avoids the 3-6s ElevenLabs WebSocket cold-start on the first call.
# on_client_connected fires before tts.start() finishes connecting its WebSocket
# (pipeline processors start concurrently), so TTSSpeakFrame would block waiting
# for the WS. Using pre-generated PCM eliminates that blocking entirely.

_GREETING = "வணக்கம்! நான் உங்கள் சேவை முகவர். நான் எப்படி உதவலாம்?"
_greeting_pcm: bytes | None = None


async def _ensure_greeting_audio() -> None:
    """Pre-generate greeting via ElevenLabs HTTP API (no WebSocket cold-start)."""
    global _greeting_pcm
    if _greeting_pcm is not None:
        return
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not voice_id or not api_key:
        logger.warning("Cannot pre-generate greeting: missing ELEVENLABS env vars")
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
                    logger.info(f"Pre-generated greeting audio: {len(_greeting_pcm)} bytes PCM")
                else:
                    text = await resp.text()
                    logger.warning(f"ElevenLabs HTTP {resp.status}: {text[:200]}")
    except Exception as e:
        logger.warning(f"Failed to pre-generate greeting audio: {e}")


class GreetingInjector(FrameProcessor):
    """Injects pre-generated PCM audio directly to transport output.

    Sits between TTS and transport.output(). When inject() is called,
    it pushes OutputAudioRawFrame chunks downstream — bypassing the
    ElevenLabs WebSocket entirely for the greeting.

    on_client_connected fires before StartFrame reaches this processor
    (pipeline processors start concurrently). inject() waits on _ready
    so frames are only pushed after this processor is fully initialized.
    """

    def __init__(self):
        super().__init__()
        self._ready = asyncio.Event()

    async def inject(self, pcm: bytes, sample_rate: int = 8000) -> None:
        # Wait until StartFrame has been received — push_frame fails before that.
        await self._ready.wait()
        chunk_size = 1600  # 100ms at 8kHz 16-bit mono
        for i in range(0, len(pcm), chunk_size):
            await self.push_frame(
                OutputAudioRawFrame(
                    audio=pcm[i : i + chunk_size],
                    sample_rate=sample_rate,
                    num_channels=1,
                )
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._ready.set()
        await self.push_frame(frame, direction)


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


async def run_bot(transport: BaseTransport, handle_sigint: bool, mode: str = "inbound"):
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
        }
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

    # Sits between TTS and transport.output() so inject() frames skip STT entirely.
    greeting_injector = GreetingInjector()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            greeting_injector,
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
        messages.append({"role": "assistant", "content": _GREETING})
        if _greeting_pcm:
            # Push pre-generated PCM directly — no ElevenLabs WS needed.
            # GreetingInjector forwards OutputAudioRawFrame straight to transport.output().
            logger.info("Playing pre-generated greeting (HTTP PCM)")
            await greeting_injector.inject(_greeting_pcm)
        else:
            # Fallback: let TTS handle it (slower due to WS cold-start)
            logger.warning("Greeting PCM not ready, falling back to TTSSpeakFrame")
            await task.queue_frames([TTSSpeakFrame(text=_GREETING)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Call disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments, mode: str = "inbound"):
    # Kick off greeting pre-generation early (runs concurrently with pipeline setup).
    # Pipeline startup takes 3+ seconds (SileroVAD load, NVIDIA gRPC, ElevenLabs WS),
    # so the HTTP call (~300-700ms) will finish well before on_client_connected fires.
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

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    await run_bot(transport, runner_args.handle_sigint, mode=mode)

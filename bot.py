import os

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
        # TTSSpeakFrame bypasses LLM entirely — audio starts in ~200ms instead of ~2s.
        # Add the greeting to the LLM context so follow-up turns have conversational history.
        greeting = "வணக்கம்! நான் உங்கள் சேவை முகவர். நான் எப்படி உதவலாம்?"
        messages.append({"role": "assistant", "content": greeting})
        await task.queue_frames([TTSSpeakFrame(text=greeting)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Call disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments, mode: str = "inbound"):
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

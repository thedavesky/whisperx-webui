import functools
import uuid
import numpy as np
from fastapi import (
    File,
    UploadFile,
)
import gradio as gr
from fastapi import APIRouter, BackgroundTasks, Depends, Request, Response, status
from typing import Any, Dict, List
from sqlalchemy.orm import Session
from datetime import datetime
from modules.whisper.data_classes import *
from modules.utils.paths import BACKEND_CACHE_DIR
from modules.whisper.whisperx_inference import WhisperXInference
from backend.common.audio import read_audio
from backend.common.config_loader import load_server_config
from backend.common.models import QueueResponse
from backend.db.task.dao import (
    add_task_to_db,
    record_task_outcome,
    update_task_status_in_db
)
from backend.db.task.models import TaskStatus, TaskType
from modules.utils.logger import get_logger


logger = get_logger()

transcription_router = APIRouter(prefix="/transcription", tags=["Transcription"])


def supplied_query_values(request: Request) -> Dict[str, Any]:
    """Query parameters the client actually sent (repeated keys become lists)."""
    grouped: Dict[str, List[str]] = {}
    for key, value in request.query_params.multi_items():
        grouped.setdefault(key, []).append(value)
    return {key: (values[0] if len(values) == 1 else values) for key, values in grouped.items()}


def with_server_defaults(param_cls, supplied: Dict[str, Any], server_section: Optional[Dict]) -> Any:
    """Build params with explicit precedence: request value > server config > model default."""
    values = param_cls().model_dump()
    for key, value in (server_section or {}).items():
        if key in values:
            values[key] = value
    for key, value in (supplied or {}).items():
        if key in values:
            values[key] = value
    return param_cls(**values)


def create_progress_callback(identifier: str):
    def progress_callback(progress_value: float):
        update_task_status_in_db(
            identifier=identifier,
            update_data={
                "uuid": identifier,
                "status": TaskStatus.IN_PROGRESS,
                "progress": round(progress_value, 2),
                "updated_at": datetime.utcnow()
            },
        )
    return progress_callback


@functools.lru_cache
def get_pipeline() -> 'WhisperXInference':
    return WhisperXInference(
        output_dir=BACKEND_CACHE_DIR
    )


def run_transcription(
    audio: np.ndarray,
    params: TranscriptionPipelineParams,
    identifier: str,
) -> List[Segment]:
    """Run one queued job and always leave the task in a truthful terminal state.

    shortcut: the job runs in this process, so a restart loses it; reconciled at
    startup, and a durable queue is required if jobs must survive restarts.
    """
    update_task_status_in_db(
        identifier=identifier,
        update_data={
            "uuid": identifier,
            "status": TaskStatus.IN_PROGRESS,
            "updated_at": datetime.utcnow()
        },
    )

    progress_callback = create_progress_callback(identifier)
    try:
        segments, elapsed_time = get_pipeline().run(
            audio,
            gr.Progress(),
            "SRT",
            False,
            progress_callback,  
            *params.to_list()
        )
        segments = [seg.model_dump() for seg in segments]

        update_task_status_in_db(
            identifier=identifier,
            update_data={
                "uuid": identifier,
                "status": TaskStatus.COMPLETED,
                "result": segments,
                "updated_at": datetime.utcnow(),
                "duration": elapsed_time,
                "progress": 1.0,
                "error": None,
            },
        )
    except Exception as exc:
        logger.exception(f"Transcription task {identifier} failed.")
        cancelled = isinstance(exc, gr.Error) and "cancelled" in str(exc).lower()
        record_task_outcome(
            identifier,
            TaskStatus.CANCELLED if cancelled else TaskStatus.FAILED,
            "Task cancelled." if cancelled else f"Transcription failed: {type(exc).__name__}",
        )
        raise
    return segments


@transcription_router.post(
    "/",
    response_model=QueueResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Transcribe Audio",
    description="Process the provided audio or video file to generate a transcription.",
)
async def transcription(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(..., description="Audio or video file to transcribe."),
    whisper_params: WhisperParams = Depends(),
    bgm_separation_params: BGMSeparationParams = Depends(),
    diarization_params: DiarizationParams = Depends(),
) -> QueueResponse:
    if not isinstance(file, np.ndarray):
        audio, info = await read_audio(file=file)
    else:
        audio, info = file, None

    supplied = supplied_query_values(request)
    server_config = load_server_config()
    params = TranscriptionPipelineParams(
        whisper=with_server_defaults(WhisperParams, supplied, server_config.get("whisper")),
        bgm_separation=with_server_defaults(
            BGMSeparationParams, supplied, server_config.get("bgm_separation")
        ),
        diarization=with_server_defaults(
            DiarizationParams, supplied, server_config.get("diarization")
        ),
    )

    identifier = add_task_to_db(
        status=TaskStatus.QUEUED,
        file_name=file.filename,
        audio_duration=info.duration if info else None,
        language=params.whisper.lang,
        task_type=TaskType.TRANSCRIPTION,
        task_params=params.to_dict(),
    )

    background_tasks.add_task(
        run_transcription,
        audio=audio,
        params=params,
        identifier=identifier,
    )

    return QueueResponse(identifier=identifier, status=TaskStatus.QUEUED, message="Transcription task has queued")

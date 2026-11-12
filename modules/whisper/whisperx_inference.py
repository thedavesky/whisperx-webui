import os
import threading
from typing import BinaryIO, Callable, Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import whisperx
from faster_whisper import available_models as faster_whisper_available_models
from transformers.models.whisper.tokenization_whisper import LANGUAGES

from modules.utils.inference_process import (
    InferenceProcess,
    WorkerCancelled,
    WorkerError,
    probe_metadata,
)
from modules.utils.logger import get_logger
from modules.utils.paths import (
    DIARIZATION_MODELS_DIR,
    OUTPUT_DIR,
    UVR_MODELS_DIR,
    WHISPER_MODELS_DIR,
)
from modules.uvr.music_separator import MusicSeparator
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.whisper.data_classes import Segment, WhisperParams


logger = get_logger()


class WhisperXInferenceEngine(BaseTranscriptionPipeline):
    def __init__(
        self,
        model_dir: str = WHISPER_MODELS_DIR,
        diarization_model_dir: str = DIARIZATION_MODELS_DIR,
        uvr_model_dir: str = UVR_MODELS_DIR,
        output_dir: str = OUTPUT_DIR,
    ):
        super().__init__(
            model_dir=model_dir,
            diarization_model_dir=diarization_model_dir,
            uvr_model_dir=uvr_model_dir,
            output_dir=output_dir,
        )

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        progress: gr.Progress = gr.Progress(),
        progress_callback: Optional[Callable] = None,
        *whisper_params,
    ) -> Tuple[List[Segment], float]:
        """Delegate to the single WhisperX ASR implementation (WhisperXWrapper)."""
        params = WhisperParams.from_list(list(whisper_params))
        wrapper = self._ensure_whisperx_wrapper()
        segments, _detected_language, elapsed_time, _device = wrapper.transcribe(
            audio,
            params,
            self.device,
            progress,
            progress_callback,
        )
        return segments, elapsed_time

    def update_model(
        self,
        model_size: str,
        compute_type: str,
        progress: gr.Progress = gr.Progress(),
        vad_options: Optional[Dict[str, float]] = None,
    ):
        progress(0, desc="Initializing Model..")
        self.current_model_size = model_size
        self.current_compute_type = compute_type
        self.model = whisperx.load_model(
            model_size,
            device=self.device,
            compute_type=compute_type,
            download_root=self.model_dir,
            vad_method="pyannote",
            vad_options=vad_options,
        )


class WhisperXInference(BaseTranscriptionPipeline):
    """Parent-side facade.

    Models are never loaded in this process: every job runs in a disposable
    worker (see modules/utils/inference_process.py) and offload() terminates
    that worker, which is the only reliable way to return model RAM/VRAM to
    the OS. Cached weights stay on SSD and the next job reloads them.
    """

    def __init__(
        self,
        model_dir: str = WHISPER_MODELS_DIR,
        diarization_model_dir: str = DIARIZATION_MODELS_DIR,
        uvr_model_dir: str = UVR_MODELS_DIR,
        output_dir: str = OUTPUT_DIR,
    ):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.diarization_model_dir = diarization_model_dir
        self.uvr_model_dir = uvr_model_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.diarization_model_dir, exist_ok=True)
        self.model = None
        self.whisperx_wrapper = None
        self.cancel_event: threading.Event | None = threading.Event()
        self._job_lock = threading.Lock()
        self._process = InferenceProcess()
        self._meta = None
        self.music_separator = MusicSeparator(
            model_dir=uvr_model_dir,
            output_dir=os.path.join(output_dir, "UVR"),
            owner=self,
        )

    def _probe(self) -> dict:
        if self._meta is None:
            self._meta = probe_metadata()
        return self._meta

    @property
    def available_models(self) -> List[str]:
        return faster_whisper_available_models()

    @property
    def available_langs(self) -> List[str]:
        return sorted(LANGUAGES.values())

    @property
    def device(self) -> str:
        return self._probe()["device"]

    @property
    def available_compute_types(self) -> List[str]:
        return self._probe()["available_compute_types"]

    @property
    def current_compute_type(self) -> str:
        return self._probe()["current_compute_type"]

    @property
    def available_diarization_devices(self) -> List[str]:
        return self._probe()["available_diarization_devices"]

    def transcribe(self, *args, **kwargs):
        raise NotImplementedError("Transcription runs in the inference worker; use run().")

    def update_model(self, *args, **kwargs):
        raise NotImplementedError("Model loading happens inside the inference worker.")

    def _run_pipeline(
        self,
        audio,
        progress=gr.Progress(),
        file_format: str = "SRT",
        add_timestamp: bool = True,
        progress_callback=None,
        *pipeline_params,
    ) -> Tuple[List[Segment], float]:
        def on_progress(value, desc):
            if progress is not None:
                progress(value, desc=desc)
            if progress_callback is not None:
                progress_callback(value)

        try:
            return self._process.request(
                {
                    "op": "pipeline",
                    "audio": audio,
                    "file_format": file_format,
                    "add_timestamp": add_timestamp,
                    "params": list(pipeline_params),
                    "dirs": {
                        "model_dir": self.model_dir,
                        "diarization_model_dir": self.diarization_model_dir,
                        "uvr_model_dir": self.uvr_model_dir,
                        "output_dir": self.output_dir,
                    },
                },
                on_progress=on_progress,
                cancel_event=self.cancel_event,
            )
        except WorkerCancelled as exc:
            raise gr.Error("Transcription cancelled") from exc
        except WorkerError as exc:
            self.offload()
            raise RuntimeError(f"Inference worker failed: {exc}") from exc

    def offload(self):
        """Stop the inference worker; the OS reclaims its RAM and VRAM on exit."""
        self._process.offload()

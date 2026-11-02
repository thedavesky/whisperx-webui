import os
import time
import tempfile
import threading
from dataclasses import replace
from typing import BinaryIO, Callable, Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import torch
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
from modules.whisper.data_classes import Segment, WhisperParams, Word


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
        start_time = time.time()
        params = WhisperParams.from_list(list(whisper_params))

        if (
            params.model_size != self.current_model_size
            or self.model is None
            or self.current_compute_type != params.compute_type
        ):
            self.update_model(
                params.model_size,
                params.compute_type,
                progress,
                vad_options=self._vad_options(params),
            )
        else:
            self._update_vad_options(params)

        if self.model is None:
            msg = "WhisperX model failed to load before transcription"
            logger.error(msg)
            raise RuntimeError(msg)

        progress(0, desc="Loading audio..")
        prepared_audio = self._prepare_audio(audio)

        self._update_transcription_options(params)

        result = self._transcribe_with_fallback(
            prepared_audio,
            params,
            progress,
        )

        segments_data = result.get("segments", [])
        detected_language = result.get("language", params.lang)
        segments_result: List[Segment] = []

        if params.word_timestamps:
            wrapper = self._ensure_whisperx_wrapper()
            segments_result = wrapper.align(
                segments=wrapper._segments_from_dict(segments_data),
                language=detected_language,
                audio=prepared_audio,
                device=self.device,
            )
        else:
            total_duration = result.get("duration") or (
                segments_data[-1]["end"] if segments_data else 0
            )
            for idx, segment in enumerate(segments_data):
                progress_n = (
                    min(segment.get("end", 0), total_duration) / total_duration
                    if total_duration
                    else 0
                )
                progress(progress_n, desc="Transcribing..")
                if progress_callback is not None:
                    progress_callback(progress_n)

                words = None
                raw_words = segment.get("words")
                if raw_words:
                    words = [
                        Word(
                            start=word.get("start"),
                            end=word.get("end"),
                            word=word.get("word"),
                            probability=word.get("score", word.get("confidence")),
                        )
                        for word in raw_words
                    ]

                segments_result.append(
                    Segment(
                        id=segment.get("id", idx),
                        seek=segment.get("seek"),
                        text=segment.get("text"),
                        start=segment.get("start"),
                        end=segment.get("end"),
                        tokens=segment.get("tokens"),
                        temperature=segment.get("temperature"),
                        avg_logprob=segment.get("avg_logprob"),
                        compression_ratio=segment.get("compression_ratio"),
                        no_speech_prob=segment.get("no_speech_prob"),
                        words=words,
                    )
                )

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def _transcribe_with_fallback(
        self,
        audio: np.ndarray,
        params: WhisperParams,
        progress: Optional[gr.Progress] = None,
    ) -> Dict[str, object]:
        chunk_size = params.chunk_length if params.chunk_length else 30
        task = "translate" if params.is_translate else "transcribe"

        batch_size = params.batch_size
        while batch_size >= 1:
            try:
                return self.model.transcribe(
                    audio,
                    batch_size=batch_size,
                    language=params.lang,
                    task=task,
                    chunk_size=chunk_size,
                )
            except RuntimeError as error:
                if self.device != "cuda" or "out of memory" not in str(error).lower():
                    raise

                next_batch_size = batch_size // 2
                if next_batch_size < 1:
                    logger.error(
                        "CUDA OOM during WhisperX transcription even at batch_size=1"
                    )
                    raise

                retry_message = (
                    "CUDA OOM during WhisperX transcription; retrying with batch_size="
                    f"{next_batch_size}"
                )
                logger.warning(retry_message)
                if progress is not None:
                    progress(0, desc=retry_message)
                torch.cuda.empty_cache()
                batch_size = next_batch_size

    @staticmethod
    def _vad_options(params: WhisperParams) -> Dict[str, float]:
        """Map the UI thresholds to the keys WhisperX expects."""
        return {
            "vad_onset": params.whisperx_vad_onset,
            "vad_offset": params.whisperx_vad_offset,
        }

    def _update_vad_options(self, params: WhisperParams) -> None:
        """Refresh the thresholds of an already loaded pipeline.

        WhisperX binarizes the VAD scores at transcription time using
        ``FasterWhisperPipeline._vad_params``, so only those values have to be
        updated when the thresholds change.
        """
        vad_params = getattr(self.model, "_vad_params", None)
        if isinstance(vad_params, dict):
            vad_params.update(self._vad_options(params))

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

    def _prepare_audio(self, audio: Union[str, BinaryIO, np.ndarray]) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            return audio

        if isinstance(audio, str):
            return whisperx.load_audio(audio)

        if hasattr(audio, "read"):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                if hasattr(audio, "seek"):
                    audio.seek(0)
                tmp.write(audio.read())
                tmp.flush()
                return whisperx.load_audio(tmp.name)

        raise ValueError("Unsupported audio input type for WhisperX inference")

    def _update_transcription_options(self, params: WhisperParams) -> None:
        if not hasattr(self.model, "options"):
            logger.warning(
                "WhisperX model does not expose transcription options; skipping override"
            )
            return

        option_overrides: Dict[
            str,
            Union[
                bool,
                float,
                int,
                Optional[str],
                List[int],
            ],
        ] = {
            "beam_size": params.beam_size,
            "patience": params.patience,
            "length_penalty": params.length_penalty,
            "initial_prompt": params.initial_prompt,
            "prefix": params.prefix,
            "suppress_blank": params.suppress_blank,
            "suppress_tokens": params.suppress_tokens,
            "hotwords": params.hotwords or None,
        }

        try:
            self.model.options = replace(self.model.options, **option_overrides)
        except TypeError as error:
            logger.error("Failed to update WhisperX transcription options: %s", error)
            raise


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
            # Live sessions call _run_pipeline directly and keep running after a
            # failed chunk, so a failed worker must not stay warm.
            self.offload()
            raise RuntimeError(f"Inference worker failed: {exc}") from exc

    def offload(self):
        """Stop the inference worker; the OS reclaims its RAM and VRAM on exit."""
        self._process.offload()

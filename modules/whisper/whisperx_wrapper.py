import gc
import importlib
import logging
import os
import time
from dataclasses import replace
import typing
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import gradio as gr
import ctranslate2
import numpy as np
import torch

from modules.utils.audio_loader import SAMPLE_RATE, load_audio
from modules.utils.logger import get_logger
from modules.whisper.data_classes import DiarizationParams, Segment, WhisperParams, Word

try:
    import whisperx  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    whisperx = None

DIARIZATION_MODEL_NAME = "pyannote/speaker-diarization-community-1"

logger = get_logger()

__all__ = ["WhisperXWrapper", "DIARIZATION_MODEL_NAME"]


class WhisperXWrapper:
    """Utility wrapper around whisperX ASR, alignment, and diarization helpers."""

    def __init__(
        self,
        model_dir: str,
        diarization_model_dir: str,
    ) -> None:
        self.model_dir = model_dir
        self.diarization_model_dir = diarization_model_dir

        self._model = None
        self._current_model_size: Optional[str] = None
        self._current_compute_type: Optional[str] = None

        self._align_model = None
        self._align_metadata = None
        self._align_language: Optional[str] = None

        self._diarization_pipeline: Optional[Any] = None
        self._diarization_device: Optional[str] = None
        self._diarization_token: Optional[str] = None
        self._diarization_compute_type: Optional[str] = None

    @staticmethod
    def _is_cuda_oom(error: RuntimeError, device: str) -> bool:
        message = str(error).lower()
        return "out of memory" in message and device.startswith("cuda")

    @staticmethod
    def _asr_available_devices() -> List[str]:
        """ASR runs on CTranslate2, which supports CUDA and CPU only."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        return devices

    @classmethod
    def _resolve_asr_device(cls, device: Optional[str]) -> str:
        available = cls._asr_available_devices()
        if device in (None, "auto"):
            return "cuda" if "cuda" in available else "cpu"
        if device not in available:
            raise gr.Error(
                f"The ASR runtime (ctranslate2) does not support device '{device}'; "
                f"available devices: {', '.join(available)}."
            )
        return device

    # ASR and forced alignment share the CTranslate2/torchaudio device set.
    _resolve_device = _resolve_asr_device

    @staticmethod
    def _available_diarization_devices() -> List[str]:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            devices.append("xpu")
        if torch.backends.mps.is_available():
            devices.append("mps")
        return devices

    @classmethod
    def _resolve_diarization_device(cls, device: Optional[str]) -> str:
        """pyannote runs on its own devices; an unavailable request is an error."""
        available = cls._available_diarization_devices()
        if device in (None, "auto"):
            return "cuda" if "cuda" in available else "cpu"
        if device not in available:
            raise gr.Error(
                f"Diarization device '{device}' is not available on this machine; "
                f"available devices: {', '.join(available)}."
            )
        return device

    @staticmethod
    def _cpu_fallback_compute_type(params: WhisperParams) -> str:
        """Pick a precision the CPU CTranslate2 runtime supports after CUDA exhaustion."""
        try:
            supported = list(ctranslate2.get_supported_compute_types("cpu"))
        except Exception:  # ctranslate2 probing can fail on exotic CPUs
            supported = ["int8", "float32"]
        requested = params.compute_type
        if requested in supported:
            return requested
        for preference in ("int8", "float32"):
            if preference in supported:
                return preference
        return supported[0]

    @staticmethod
    def _available_diarization_compute_types(device: str) -> List[str]:
        """pyannote runs fp32; the diarization pipeline has no lower-precision path."""
        return ["float32"]

    @staticmethod
    def _supported_alignment_languages() -> Set[str]:
        align_module = importlib.import_module("whisperx.alignment")
        return set(align_module.DEFAULT_ALIGN_MODELS_TORCH) | set(align_module.DEFAULT_ALIGN_MODELS_HF)

    @staticmethod
    def _require_whisperx() -> None:
        if whisperx is None:
            raise ImportError(
                "whisperx is required for alignment/diarization. Install it with `pip install whisperx`."
            )

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
        vad_params = getattr(self._model, "_vad_params", None)
        if isinstance(vad_params, dict):
            vad_params.update(self._vad_options(params))

    def _load_model(self, params: WhisperParams, device: str) -> None:
        self._require_whisperx()
        compute_type = params.compute_type or "int8"
        if (
            self._model is None
            or self._current_model_size != params.model_size
            or self._current_compute_type != compute_type
        ):
            logger.info(
                "Loading WhisperX model %s (compute=%s) on %s", params.model_size, compute_type, device
            )
            self._model = whisperx.load_model(
                params.model_size,
                device=device,
                compute_type=compute_type,
                download_root=self.model_dir,
                vad_method="pyannote",
                vad_options=self._vad_options(params),
            )
            self._current_model_size = params.model_size
            self._current_compute_type = compute_type

    def _update_asr_options(self, params: WhisperParams) -> None:
        """Mirror whisper parameters onto the whisperx transcription options."""

        if not hasattr(self._model, "options"):
            return

        option_updates = {
            "beam_size": params.beam_size,
            "best_of": params.best_of,
            "patience": params.patience,
            "length_penalty": params.length_penalty,
            "repetition_penalty": params.repetition_penalty,
            "no_repeat_ngram_size": params.no_repeat_ngram_size,
            "log_prob_threshold": params.log_prob_threshold,
            "no_speech_threshold": params.no_speech_threshold,
            "compression_ratio_threshold": params.compression_ratio_threshold,
            "condition_on_previous_text": params.condition_on_previous_text,
            "prompt_reset_on_temperature": params.prompt_reset_on_temperature,
            "temperatures": [params.temperature],
            "initial_prompt": params.initial_prompt,
            "prefix": params.prefix,
            "suppress_blank": params.suppress_blank,
            "suppress_tokens": params.suppress_tokens,
            "max_initial_timestamp": params.max_initial_timestamp,
            "prepend_punctuations": params.prepend_punctuations,
            "append_punctuations": params.append_punctuations,
            "max_new_tokens": params.max_new_tokens,
            "hallucination_silence_threshold": params.hallucination_silence_threshold,
            "hotwords": params.hotwords,
            "without_timestamps": False,
            "word_timestamps": False,
        }

        self._model.options = replace(self._model.options, **option_updates)

    @staticmethod
    def _reject_unsupported_options(params: WhisperParams) -> None:
        """Fail loudly on options the WhisperX ASR pipeline cannot apply."""
        if params.word_timestamps and params.is_translate:
            raise gr.Error(
                "Word timestamps cannot be aligned onto translated text. Transcription "
                "with translation still exports segment-level subtitles; turn off word "
                "timestamps (or translation) to continue."
            )

        defaults = WhisperParams()
        unsupported = [
            name
            for name in ("language_detection_threshold", "language_detection_segments")
            if getattr(params, name) != getattr(defaults, name)
        ]
        if unsupported:
            names = ", ".join(unsupported)
            raise gr.Error(
                f"The WhisperX pipeline does not use {names}; keep them at their defaults "
                f"({', '.join(str(getattr(defaults, name)) for name in unsupported)})."
            )

    @staticmethod
    def _segments_from_dict(segments: Sequence[dict], prefix_speaker: bool = False) -> List[Segment]:
        segment_models: List[Segment] = []
        for data in segments:
            words_data = data.get("words") or []
            words: List[Word] = []
            for word in words_data:
                score = word.get("score")
                words.append(
                    Word(
                        start=word.get("start"),
                        end=word.get("end"),
                        word=word.get("word") or word.get("text"),
                        probability=score if score is not None else word.get("probability"),
                        speaker=word.get("speaker"),
                    )
                )

            text = data.get("text")
            speaker = data.get("speaker")
            if prefix_speaker and speaker:
                # Maintain backward compatibility with existing diarization outputs.
                text = f"{speaker}|{text.strip()}" if text else text

            segment_models.append(
                Segment(
                    id=data.get("id"),
                    seek=data.get("seek"),
                    text=text,
                    start=data.get("start"),
                    end=data.get("end"),
                    tokens=data.get("tokens"),
                    temperature=data.get("temperature"),
                    avg_logprob=data.get("avg_logprob"),
                    compression_ratio=data.get("compression_ratio"),
                    no_speech_prob=data.get("no_speech_prob"),
                    speaker=speaker,
                    words=words or None,
                )
            )
        return segment_models

    @staticmethod
    def _segments_to_dict(segments: Sequence[Segment]) -> List[dict]:
        serialized = []
        for segment in segments:
            data = segment.model_dump(exclude_none=True)
            if segment.words:
                data["words"] = [word.model_dump(exclude_none=True) for word in segment.words]
            serialized.append(data)
        return serialized

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        params: WhisperParams,
        device: str,
        progress: gr.Progress,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[List[Segment], Optional[str], float, str]:
        """Transcribe audio with WhisperX.

        Returns the segments, the detected language, the elapsed time, and the
        device that actually ran the model (it changes on CPU fallback), so later
        stages never return to an exhausted GPU.
        """
        self._require_whisperx()
        self._reject_unsupported_options(params)
        resolved_device = self._resolve_device(device)
        active_params = params
        try:
            self._load_model(active_params, resolved_device)
        except RuntimeError as exc:
            if not self._is_cuda_oom(exc, resolved_device):
                raise
            # Release the half-loaded GPU model before falling back.
            self._model = None
            self._current_model_size = None
            self._free_memory()
            resolved_device = "cpu"
            active_params = params.model_copy(
                update={
                    "compute_type": self._cpu_fallback_compute_type(params),
                    "batch_size": max(1, params.batch_size // 2),
                }
            )
            progress(0, desc="CUDA out of memory while loading the model; retrying on CPU..")
            self._load_model(active_params, resolved_device)
        self._update_vad_options(active_params)
        self._update_asr_options(active_params)

        start_time = time.time()
        audio_array = load_audio(audio)
        progress(0, desc="Transcribing with WhisperX..")

        task = "translate" if active_params.is_translate else "transcribe"
        batch_size = max(1, active_params.batch_size)
        while True:
            try:
                result = self._model.transcribe(
                    audio_array,
                    batch_size=batch_size,
                    language=active_params.lang,
                    task=task,
                    chunk_size=active_params.chunk_length,
                )
                params.batch_size = batch_size
                params.compute_type = active_params.compute_type
                break
            except RuntimeError as exc:
                if not self._is_cuda_oom(exc, resolved_device):
                    raise

                logger.warning(
                    "CUDA OOM during WhisperX transcription with batch size %s on %s; retrying.",
                    batch_size,
                    resolved_device,
                    exc_info=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                reduced_batch_size = max(1, batch_size // 2)
                if reduced_batch_size < batch_size:
                    batch_size = reduced_batch_size
                    active_params = active_params.model_copy(update={"batch_size": batch_size})
                    progress(0, desc=f"Retrying WhisperX with batch size {batch_size}..")
                    continue

                if resolved_device == "cpu":
                    raise

                # Release the model that exhausted the GPU before reloading on CPU.
                self._model = None
                self._current_model_size = None
                self._free_memory()
                resolved_device = "cpu"
                active_params = params.model_copy(
                    update={
                        "compute_type": self._cpu_fallback_compute_type(params),
                        "batch_size": batch_size,
                    }
                )
                self._load_model(active_params, resolved_device)
                self._update_asr_options(active_params)
                progress(0, desc="Retrying WhisperX on CPU after CUDA OOM..")

        language = result.get("language") or active_params.lang
        segments_data = result.get("segments", [])
        audio_duration = max(len(audio_array) / SAMPLE_RATE, 1e-6)

        segments = []
        for segment in segments_data:
            progress_value = min(segment.get("end", 0.0) / audio_duration, 0.99)
            progress(progress_value, desc="Transcribing with WhisperX..")
            if progress_callback is not None:
                progress_callback(progress_value)
        segments = self._segments_from_dict(segments_data)

        elapsed = time.time() - start_time
        return segments, language, elapsed, resolved_device

    def align(
        self,
        segments: Sequence[Segment],
        language: Optional[str],
        audio: Union[str, np.ndarray],
        device: str,
    ) -> List[Segment]:
        """Align segments using WhisperX forced alignment."""
        if not segments:
            return list(segments)

        self._require_whisperx()
        resolved_device = self._resolve_device(device)
        language_code = (language or "en").lower()
        if language_code not in self._supported_alignment_languages():
            raise gr.Error(
                f"WhisperX has no alignment model for language '{language_code}', so word "
                f"timestamps cannot be produced. Disable word timestamps or pick a supported language."
            )

        if self._align_model is None or self._align_language != language_code:
            logger.info("Loading WhisperX alignment model for language %s", language_code)
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language_code,
                device=resolved_device,
                model_dir=self.model_dir,
            )
            self._align_language = language_code

        audio_array = load_audio(audio)
        aligned_result = whisperx.align(
            self._segments_to_dict(segments),
            self._align_model,
            self._align_metadata,
            audio_array,
            resolved_device,
            return_char_alignments=False,
        )
        return self._segments_from_dict(aligned_result.get("segments", []))

    def _load_diarization_pipeline(
        self,
        diarize_module: Any,
        device: str,
        token: Optional[str],
    ) -> Any:
        """Instantiate WhisperX's own diarization pipeline."""
        os.makedirs(self.diarization_model_dir, exist_ok=True)

        if not os.listdir(self.diarization_model_dir) and not token:
            raise RuntimeError(
                "Failed to diarize. You need a Hugging Face token and must accept the model terms to download the diarization pipeline. "
                f"Visit https://huggingface.co/{DIARIZATION_MODEL_NAME} to grant access, then provide the token."
            )

        train_logger = logging.getLogger("speechbrain.utils.train_logger")
        train_logger.disabled = True
        try:
            return diarize_module.DiarizationPipeline(
                model_name=DIARIZATION_MODEL_NAME,
                device=device,
                token=token,
                cache_dir=self.diarization_model_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize the diarization pipeline. "
                "Ensure the Hugging Face model access is granted and dependencies match the trained versions."
            ) from exc
        finally:
            train_logger.disabled = False

    def diarize(
        self,
        audio: Union[str, np.ndarray],
        segments: Sequence[Segment],
        diarization_params: DiarizationParams,
    ) -> List[Segment]:
        """Run WhisperX diarization and merge speakers into segments and words."""
        if not segments:
            return list(segments)

        self._require_whisperx()
        diarize_module = importlib.import_module("whisperx.diarize")

        resolved_device = self._resolve_diarization_device(
            diarization_params.diarization_device)
        token = diarization_params.hf_token or os.environ.get("HF_TOKEN")
        available_compute_types = self._available_diarization_compute_types(
            resolved_device)
        requested_compute_type = diarization_params.diarization_compute_type
        if requested_compute_type not in available_compute_types:
            if requested_compute_type is not None:
                logger.warning(
                    "Diarization pipeline does not support compute_type=%s; running with %s.",
                    requested_compute_type,
                    available_compute_types[0],
                )
            requested_compute_type = available_compute_types[0]

        if (
            self._diarization_pipeline is None
            or self._diarization_device != resolved_device
            or self._diarization_token != token
            or self._diarization_compute_type != requested_compute_type
        ):
            self._diarization_pipeline = self._load_diarization_pipeline(
                diarize_module=diarize_module,
                device=resolved_device,
                token=token,
            )
            self._diarization_device = resolved_device
            self._diarization_token = token
            self._diarization_compute_type = requested_compute_type

        diarize_df = self._diarization_pipeline(audio)

        tag_word_speakers = bool(
            getattr(diarization_params, "assign_word_speakers", False))
        fill_nearest_speaker = bool(
            getattr(diarization_params, "fill_nearest_speaker", False))

        # WhisperX tags every word present in the payload, so the words are withheld
        # to keep word-level speaker labels opt-in.
        transcript_payload: Dict[str, List[dict]] = {"segments": []}
        for segment_data in self._segments_to_dict(segments):
            if not tag_word_speakers:
                segment_data.pop("words", None)
            transcript_payload["segments"].append(segment_data)

        diarized = diarize_module.assign_word_speakers(
            diarize_df,
            transcript_payload,
            fill_nearest=fill_nearest_speaker,
        )
        segments_result = self._segments_from_dict(
            diarized.get("segments", []),
            prefix_speaker=True,
        )

        if not tag_word_speakers:
            for segment_model, original in zip(segments_result, segments):
                if original.words:
                    segment_model.words = [word.model_copy()
                                           for word in original.words]

        return segments_result

    @staticmethod
    def _free_memory() -> None:
        gc.collect()

        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    def offload_asr(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._free_memory()

    def offload_alignment(self) -> None:
        if self._align_model is not None:
            del self._align_model
            self._align_model = None
            self._align_metadata = None
            self._align_language = None
            self._free_memory()

    def offload_diarizer(self) -> None:
        if self._diarization_pipeline is not None:
            del self._diarization_pipeline
            self._diarization_pipeline = None
            self._diarization_device = None
            self._diarization_token = None
            self._diarization_compute_type = None
            self._free_memory()

    def offload_all(self) -> None:
        self.offload_asr()
        self.offload_alignment()
        self.offload_diarizer()

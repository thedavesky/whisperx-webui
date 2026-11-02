from typing import Optional, Union, List, Dict
import numpy as np
import torchaudio
import soundfile as sf
import os
import threading
import torch
import gc
import gradio as gr
from datetime import datetime
import traceback

from modules.utils.paths import DEFAULT_PARAMETERS_CONFIG_PATH, UVR_MODELS_DIR, UVR_OUTPUT_DIR
from modules.utils.files_manager import load_yaml, save_yaml, is_video
from modules.utils.audio_loader import load_audio
from modules.utils.inference_process import InferenceProcess, WorkerError, probe_metadata
from modules.utils.logger import get_logger

logger = get_logger()

UVR_AVAILABLE_MODELS = ["UVR-MDX-NET-Inst_HQ_4", "UVR-MDX-NET-Inst_3"]

try:
    from uvr.models import MDX, Demucs, VrNetwork, MDXC
except Exception as e:
    logger.warning(
        "Failed to import uvr. BGM separation feature will not work. "
        "Please open an issue on GitHub if you encounter this error. "
        f"Error: {type(e).__name__}: {traceback.format_exc()}"
    )


class MusicSeparatorEngine:
    def __init__(self,
                 model_dir: Optional[str] = UVR_MODELS_DIR,
                 output_dir: Optional[str] = UVR_OUTPUT_DIR):
        self.model = None
        self.device = self.get_device()
        self.available_devices = ["cpu", "cuda", "xpu", "mps"]
        self.model_dir = model_dir
        self.output_dir = output_dir
        instrumental_output_dir = os.path.join(self.output_dir, "instrumental")
        vocals_output_dir = os.path.join(self.output_dir, "vocals")
        os.makedirs(instrumental_output_dir, exist_ok=True)
        os.makedirs(vocals_output_dir, exist_ok=True)
        self.audio_info = None
        self.available_models = UVR_AVAILABLE_MODELS
        self.default_model = self.available_models[0]
        self.current_model_size = self.default_model
        self.model_config = {
            "segment": 256,
            "split": True
        }
        self._job_lock = threading.Lock()

    def update_model(self,
                     model_name: str = "UVR-MDX-NET-Inst_1",
                     device: Optional[str] = None,
                     segment_size: int = 256):
        """
        Update model with the given model name

        Args:
            model_name (str): Model name.
            device (str): Device to use for the model.
            segment_size (int): Segment size for the prediction.
        """
        if device is None:
            device = self.device

        self.device = device
        self.model_config = {
            "segment": segment_size,
            "split": True
        }
        self.model = MDX(name=model_name,
                         other_metadata=self.model_config,
                         device=self.device,
                         logger=None,
                         model_dir=self.model_dir)

    def _separate(self,
                  audio: Union[str, np.ndarray],
                  model_name: str,
                  device: Optional[str] = None,
                  segment_size: int = 256,
                  save_file: bool = False,
                  progress: gr.Progress = gr.Progress()) -> tuple[np.ndarray, np.ndarray, List]:
        """
        Separate the background music from the audio.

        Args:
            audio (Union[str, np.ndarray]): Audio path or numpy array.
            model_name (str): Model name.
            device (str): Device to use for the model.
            segment_size (int): Segment size for the prediction.
            save_file (bool): Whether to save the separated audio to output path or not.
            progress (gr.Progress): Gradio progress indicator.

        Returns:
            A Tuple of
            np.ndarray: Instrumental numpy arrays.
            np.ndarray: Vocals numpy arrays.
            file_paths: List of file paths where the separated audio is saved. Return empty when save_file is False.
        """
        if isinstance(audio, str):
            output_filename, ext = os.path.basename(audio), ".wav"
            output_filename, orig_ext = os.path.splitext(output_filename)

            if is_video(audio):
                audio = load_audio(audio)
                sample_rate = 16000
            else:
                self.audio_info = torchaudio.info(audio)
                sample_rate = self.audio_info.sample_rate
        else:
            timestamp = datetime.now().strftime("%m%d%H%M%S")
            output_filename, ext = f"UVR-{timestamp}", ".wav"
            sample_rate = 16000

        model_config = {
            "segment": segment_size,
            "split": True
        }

        if (self.model is None or
                self.current_model_size != model_name or
                self.model_config != model_config or
                self.model.sample_rate != sample_rate or
                self.device != device):
            progress(0, desc="Initializing UVR Model..")
            self.update_model(
                model_name=model_name,
                device=device,
                segment_size=segment_size
            )
            self.model.sample_rate = sample_rate

        progress(0, desc="Separating background music from the audio.. "
                         "(It will only display 0% until the job is complete.) ")
        result = self.model(audio)
        instrumental, vocals = result["instrumental"].T, result["vocals"].T

        file_paths = []
        if save_file:
            instrumental_output_path = os.path.join(self.output_dir, "instrumental", f"{output_filename}-instrumental{ext}")
            vocals_output_path = os.path.join(self.output_dir, "vocals", f"{output_filename}-vocals{ext}")
            sf.write(instrumental_output_path, instrumental, sample_rate, format="WAV")
            sf.write(vocals_output_path, vocals, sample_rate, format="WAV")
            file_paths += [instrumental_output_path, vocals_output_path]

        return instrumental, vocals, file_paths

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.xpu.is_available():
            return "xpu"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def offload(self):
        """Offload the model and free up the memory"""
        if self.model is not None:
            del self.model
            self.model = None

        gc.collect()

        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        if hasattr(torch, "xpu") and torch.xpu.is_initialized():
            torch.xpu.empty_cache()
        self.audio_info = None

    @staticmethod
    def cache_parameters(model_size: str,
                         segment_size: int):
        cached_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        cached_uvr_params = cached_params["bgm_separation"]
        uvr_params_to_cache = {
            "uvr_model_size": model_size,
            "segment_size": segment_size
        }
        cached_uvr_params = {**cached_uvr_params, **uvr_params_to_cache}
        cached_params["bgm_separation"] = cached_uvr_params
        save_yaml(cached_params, DEFAULT_PARAMETERS_CONFIG_PATH)


class MusicSeparator:
    """Parent-side client. Separation runs inside a disposable worker process
    so that offload() can return the model's RAM and VRAM to the OS."""

    def __init__(self,
                 model_dir: Optional[str] = UVR_MODELS_DIR,
                 output_dir: Optional[str] = UVR_OUTPUT_DIR,
                 owner=None):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.available_models = UVR_AVAILABLE_MODELS
        self.available_devices = ["cpu", "cuda", "xpu", "mps"]
        self._owner = owner
        self._own_process = None if owner is not None else InferenceProcess()
        self._own_lock = threading.Lock()
        self._meta = None

    @property
    def device(self) -> str:
        return self._probe()["uvr_device"]

    @property
    def _job_lock(self) -> threading.Lock:
        return self._owner._job_lock if self._owner is not None else self._own_lock

    def _probe(self) -> Dict:
        if self._meta is None:
            self._meta = self._owner._probe() if self._owner is not None else probe_metadata()
        return self._meta

    def offload(self):
        """Stop the worker(s); the OS reclaims the model memory when they exit."""
        if self._owner is not None:
            self._owner.offload()
        else:
            self._own_process.offload()

    def _run(self, message: Dict, progress) -> object:
        process = self._owner._process if self._owner is not None else self._own_process
        try:
            return process.request(
                message,
                on_progress=lambda value, desc: progress(value, desc=desc),
            )
        except WorkerError as exc:
            raise RuntimeError(f"Error separating audio: {exc}") from exc

    def separate(self,
                 audio: Union[str, np.ndarray],
                 model_name: str,
                 device: Optional[str] = None,
                 segment_size: int = 256,
                 save_file: bool = False,
                 progress: gr.Progress = gr.Progress()) -> tuple[np.ndarray, np.ndarray, List]:
        with self._job_lock:
            try:
                return self._run(
                    {
                        "op": "separate",
                        "dirs": {"model_dir": self.model_dir, "output_dir": self.output_dir},
                        "audio": audio,
                        "model_name": model_name,
                        "device": device,
                        "segment_size": segment_size,
                        "save_file": save_file,
                    },
                    progress,
                )
            finally:
                self.offload()

    def separate_files(self,
                       files: List,
                       model_name: str,
                       device: Optional[str] = None,
                       segment_size: int = 256,
                       save_file: bool = True,
                       progress: gr.Progress = gr.Progress()) -> List[str]:
        """Separate the background music from the audio files. Returns only last Instrumental and vocals file paths
        to display into gr.Audio()"""
        MusicSeparatorEngine.cache_parameters(model_size=model_name, segment_size=segment_size)

        with self._job_lock:
            try:
                return self._run(
                    {
                        "op": "separate_batch",
                        "dirs": {"model_dir": self.model_dir, "output_dir": self.output_dir},
                        "files": list(files),
                        "model_name": model_name,
                        "device": device,
                        "segment_size": segment_size,
                        "save_file": save_file,
                    },
                    progress,
                )
            finally:
                self.offload()

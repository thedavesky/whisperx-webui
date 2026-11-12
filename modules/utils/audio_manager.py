from typing import Optional, Union
import os

import gradio as gr
import numpy as np

from modules.utils.logger import get_logger

logger = get_logger()


def validate_audio(audio: Optional[Union[str, np.ndarray]] = None) -> None:
    """Raise a gr.Error for missing, empty, or unprovided inputs.

    Decode/corruption errors are raised where the audio is actually loaded
    (modules.utils.audio_loader.load_audio), so files are never decoded twice.
    """
    if audio is None:
        raise gr.Error("No audio input was provided. Upload a file, use the Youtube tab, or record with the mic.")
    if isinstance(audio, np.ndarray):
        if audio.size == 0:
            raise gr.Error("The provided audio is empty; there is nothing to transcribe.")
        return
    if not isinstance(audio, str):
        raise gr.Error(f"Unsupported audio input of type {type(audio).__name__}; use a file path or an array.")
    if not os.path.exists(audio):
        raise gr.Error(f"The file {audio} does not exist. Please check the path.")
    if os.path.getsize(audio) == 0:
        raise gr.Error(f"The file {audio} is empty (0 bytes).")

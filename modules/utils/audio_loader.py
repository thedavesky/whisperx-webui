# Adapted from https://github.com/m-bain/whisperX/blob/main/whisperx/audio.py

import os
import subprocess
import tempfile
from typing import Union

import numpy as np
from scipy.io.wavfile import write

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000


def load_audio(file: Union[str, np.ndarray], sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Open an audio file or process a numpy array containing audio data as mono waveform, resampling as necessary.

    Parameters
    ----------
    file: Union[str, np.ndarray]
        The audio file to open or a numpy array containing the audio data.

    sr: int
        The sample rate to resample the audio if necessary.

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    if isinstance(file, np.ndarray):
        if file.dtype != np.float32:
            file = file.astype(np.float32)
        if file.ndim > 1:
            file = np.mean(file, axis=1)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(temp_file.name, SAMPLE_RATE, (file * 32768).astype(np.int16))
        temp_file_path = temp_file.name
        temp_file.close()
    else:
        temp_file_path = file

    try:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            temp_file_path,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    finally:
        if isinstance(file, np.ndarray):
            os.remove(temp_file_path)

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


__all__ = ["SAMPLE_RATE", "load_audio"]

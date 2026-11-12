# Adapted from https://github.com/m-bain/whisperX/blob/main/whisperx/audio.py

import subprocess
from typing import Union

import numpy as np

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000


def load_audio(file: Union[str, np.ndarray], sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Open an audio file or normalize a numpy array into a mono 16 kHz float32 waveform.

    Parameters
    ----------
    file: Union[str, np.ndarray]
        The audio file to open (decoded with ffmpeg at any sample rate) or a numpy
        array containing the audio data. Raw arrays carry no sample rate, so they
        must already be at ``sr`` (16 kHz by default).
    sr: int
        Sample rate of the array input; file inputs are always resampled to it.

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype in [-1.0, 1.0].
    """
    if isinstance(file, np.ndarray):
        audio = np.asarray(file, dtype=np.float32)
        audio = np.squeeze(audio)
        if audio.ndim > 1:
            axis = 0 if audio.shape[0] < audio.shape[1] else 1
            audio = np.mean(audio, axis=axis)
        audio = audio.reshape(-1).astype(np.float32, copy=False)
        if not np.all(np.isfinite(audio)):
            raise ValueError("Audio contains non-finite (NaN/inf) samples.")
        if audio.size and float(np.max(np.abs(audio))) > 1.0:
            audio = np.clip(audio, -1.0, 1.0)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Array audio must already be at {SAMPLE_RATE} Hz, got {sr} Hz.")
        return audio

    try:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
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

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


__all__ = ["SAMPLE_RATE", "load_audio"]

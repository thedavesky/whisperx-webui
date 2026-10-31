"""Compatibility shims required by the upstream WhisperX/pyannote diarization stack.

`pyannote.audio` 3.x still calls `huggingface_hub.hf_hub_download` with the removed
`use_auth_token` keyword, and it also relies on `torchaudio` helpers that recent
torchaudio releases moved or dropped. These helpers keep those call sites working
without patching the third-party sources.

They are applied lazily by `WhisperXWrapper` before `whisperx.diarize` (and therefore
`pyannote.audio`) is first imported, which matters because `pyannote` binds
`hf_hub_download` directly with `from huggingface_hub import hf_hub_download`.
"""

import sys
from dataclasses import dataclass
from functools import wraps
from typing import List

try:
    import torchaudio
except ImportError:  # pragma: no cover - torchaudio is expected in production installs
    torchaudio = None

_PATCHED_HF_HUB_DOWNLOAD = None

__all__ = [
    "ensure_diarization_compatibility",
    "patch_huggingface_hub_auth_token",
    "patch_torchaudio_metadata_aliases",
]


def _rename_auth_token_kwarg(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        if "use_auth_token" in kwargs:
            token = kwargs.pop("use_auth_token")
            if token is not None:
                kwargs.setdefault("token", token)
        return func(*args, **kwargs)

    return _wrapper


def patch_huggingface_hub_auth_token() -> None:
    """Rename pyannote's deprecated `use_auth_token` kwarg to huggingface_hub's `token`."""
    global _PATCHED_HF_HUB_DOWNLOAD

    import huggingface_hub

    if _PATCHED_HF_HUB_DOWNLOAD is None:
        _PATCHED_HF_HUB_DOWNLOAD = _rename_auth_token_kwarg(huggingface_hub.hf_hub_download)

    huggingface_hub.hf_hub_download = _PATCHED_HF_HUB_DOWNLOAD

    # Modules such as `pyannote.audio.core.pipeline` and `pyannote.audio.core.model`
    # import `hf_hub_download` directly, so any already-imported module holding the
    # unpatched reference has to be rebound as well.
    for module in list(sys.modules.values()):
        if module is None:
            continue
        # Only look at real module attributes; some packages implement a lazy
        # module-level ``__getattr__`` that would do work on plain attribute access.
        reference = getattr(module, "__dict__", {}).get("hf_hub_download")
        if reference is not None and reference is not _PATCHED_HF_HUB_DOWNLOAD and callable(reference):
            setattr(module, "hf_hub_download", _PATCHED_HF_HUB_DOWNLOAD)


def patch_torchaudio_metadata_aliases() -> None:
    """Restore the torchaudio helpers that pyannote expects on newer releases."""
    if torchaudio is None:
        return

    if not hasattr(torchaudio, "AudioMetaData"):
        io_module = getattr(torchaudio, "io", None)
        io_meta = getattr(io_module, "AudioMetaData", None) if io_module else None
        if io_meta is not None:
            torchaudio.AudioMetaData = io_meta
        else:  # pragma: no cover - fallback for torchaudio>=2.9 where the class was removed

            @dataclass
            class _AudioMetaData:  # minimal signature used by pyannote.audio type hints
                sample_rate: int
                num_channels: int
                num_frames: int
                bits_per_sample: int
                encoding: str

            torchaudio.AudioMetaData = _AudioMetaData

    if not hasattr(torchaudio, "list_audio_backends"):

        def _list_audio_backends() -> List[str]:  # pragma: no cover - shim for torchaudio>=2.9
            return ["soundfile"]

        torchaudio.list_audio_backends = _list_audio_backends


def ensure_diarization_compatibility() -> None:
    """Apply every shim the upstream diarization pipeline needs. Safe to call repeatedly."""
    patch_huggingface_hub_auth_token()
    patch_torchaudio_metadata_aliases()

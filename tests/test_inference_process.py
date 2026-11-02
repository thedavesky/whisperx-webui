import queue
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.utils.inference_process import (
    InferenceProcess,
    WorkerCancelled,
    WorkerError,
    probe_metadata,
)


def test_probe_returns_metadata():
    meta = probe_metadata()
    assert meta["device"] in ("cpu", "cuda", "xpu", "mps")
    assert isinstance(meta["available_compute_types"], list)
    assert meta["uvr_available_models"]
    assert meta["current_compute_type"] in meta["available_compute_types"]


def test_offload_terminates_worker_and_is_idempotent():
    process = InferenceProcess()
    try:
        assert process.request({"op": "probe"})["device"]
        proc = process._proc
        process.offload()
        process.offload()
        assert process._proc is None
        assert proc.poll() is not None
        # A new request after offload spawns a fresh worker.
        assert process.request({"op": "probe"})["device"]
    finally:
        process.offload()


def test_worker_error_does_not_kill_worker():
    process = InferenceProcess()
    try:
        process.request({"op": "probe"})
        proc = process._proc
        with pytest.raises(WorkerError):
            process.request({"op": "does-not-exist"})
        assert process._proc is proc
        assert process.request({"op": "probe"})["device"]
    finally:
        process.offload()


def test_cancel_checked_even_with_queued_responses(monkeypatch):
    process = InferenceProcess()
    monkeypatch.setattr(process, "_ensure_started", lambda: None)
    monkeypatch.setattr(process, "_send", lambda message: None)
    process._responses = queue.Queue()
    process._responses.put({"type": "progress", "value": 0.5})
    process._responses.put({"type": "result", "payload": "completed"})

    cancel_event = threading.Event()
    with pytest.raises(WorkerCancelled):
        process.request(
            {"op": "probe"},
            on_progress=lambda value, desc: cancel_event.set(),
            cancel_event=cancel_event,
        )


def test_already_cancelled_request_does_not_spawn_worker():
    process = InferenceProcess()
    cancel_event = threading.Event()
    cancel_event.set()
    with pytest.raises(WorkerCancelled):
        process.request({"op": "probe"}, cancel_event=cancel_event)
    assert process._proc is None


def test_facade_starts_no_worker_until_used():
    from modules.whisper.whisperx_inference import WhisperXInference

    inferencer = WhisperXInference()
    try:
        assert inferencer._process._proc is None
        assert inferencer.device in ("cpu", "cuda", "xpu", "mps")
        # Probing happens in a short-lived process, the facade owns no worker.
        assert inferencer._process._proc is None
        inferencer.offload()
    finally:
        inferencer.offload()


def test_pipeline_failure_disposes_worker(monkeypatch):
    from modules.whisper.whisperx_inference import WhisperXInference

    inferencer = WhisperXInference()
    try:
        assert inferencer._process.request({"op": "probe"})["device"]
        proc = inferencer._process._proc

        def failing_request(message, on_progress=None, cancel_event=None):
            raise WorkerError("simulated failure")

        monkeypatch.setattr(inferencer._process, "request", failing_request)
        with pytest.raises(RuntimeError, match="simulated failure"):
            inferencer._run_pipeline("unused.wav")
        assert inferencer._process._proc is None
        assert proc.poll() is not None
    finally:
        inferencer.offload()

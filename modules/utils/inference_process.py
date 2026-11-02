import atexit
import os
import pickle
import queue
import subprocess
import sys
import threading

from modules.utils.paths import WEBUI_DIR

_GRACEFUL_SHUTDOWN_TIMEOUT = 5.0
_FORCE_SHUTDOWN_TIMEOUT = 5.0
_POLL_INTERVAL = 0.2

_CHILD_CODE = (
    f"import sys; sys.path.insert(0, {WEBUI_DIR!r}); "
    "import modules.utils.inference_process as _ip; _ip.child_main()"
)

_live_processes: set = set()
_live_lock = threading.Lock()


class WorkerError(RuntimeError):
    """An inference job failed inside the worker process."""


class WorkerCancelled(Exception):
    """Cancellation was requested while the worker was running."""


class WorkerDied(RuntimeError):
    """The worker process exited or could not be reached."""


class InferenceProcess:
    """Owns one disposable inference child; offload() ends it, requests re-spawn it."""

    def __init__(self) -> None:
        self._proc = None
        self._responses = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _ensure_started(self) -> None:
        if self.running:
            return
        creationflags = getattr(
            subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        env = os.environ.copy()
        # shortcut: whisperx/pyannote checkpoints predate torch 2.6's weights_only
        # default; trust the pinned model files like the app's allowlist shim did.
        # Upgrade path: register exact safe globals once upstream checkpoints ship fixes.
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _CHILD_CODE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            creationflags=creationflags,
        )
        self._responses = queue.Queue()
        threading.Thread(
            target=self._read_responses,
            args=(self._proc.stdout, self._responses),
            daemon=True,
        ).start()
        with _live_lock:
            _live_processes.add(self)

    @staticmethod
    def _read_responses(stream, out: "queue.Queue") -> None:
        while True:
            try:
                message = pickle.load(stream)
            except (EOFError, OSError, pickle.UnpicklingError, ValueError):
                out.put(None)
                return
            out.put(message)

    def _send(self, message: dict) -> None:
        try:
            pickle.dump(message, self._proc.stdin)
            self._proc.stdin.flush()
        except (OSError, ValueError, pickle.PicklingError) as exc:
            self.offload()
            raise WorkerDied(
                f"Could not reach the inference worker: {exc}") from exc

    def request(self, message: dict, on_progress=None, cancel_event=None):
        """Send one op, pump progress messages, and return the result payload."""
        if cancel_event is not None and cancel_event.is_set():
            self.offload()
            raise WorkerCancelled()
        self._ensure_started()
        self._send(message)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                self.offload()
                raise WorkerCancelled()
            try:
                response = self._responses.get(timeout=_POLL_INTERVAL)
            except queue.Empty:
                continue
            if response is None:
                exit_code = self._proc.poll() if self._proc is not None else None
                self.offload()
                raise WorkerDied(
                    f"The inference worker exited unexpectedly (exit code {exit_code})."
                )
            kind = response.get("type")
            if kind == "progress":
                if on_progress is not None:
                    on_progress(response.get("value"), response.get("desc"))
            elif kind == "result":
                return response.get("payload")
            else:
                raise WorkerError(response.get(
                    "message", "Inference worker failed"))

    def offload(self) -> None:
        """Terminate the worker so the OS reclaims its RAM and VRAM. Idempotent."""
        process = self._proc
        if process is None:
            return
        self._proc = None
        try:
            pickle.dump({"op": "shutdown"}, process.stdin)
            process.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            process.wait(timeout=_GRACEFUL_SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=_FORCE_SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for pipe in (process.stdin, process.stdout):
            try:
                pipe.close()
            except OSError:
                pass
        with _live_lock:
            _live_processes.discard(self)


def probe_metadata() -> dict:
    """Device capability metadata, computed in a short-lived worker so the
    parent process never initializes CUDA just to fill UI dropdowns."""
    process = InferenceProcess()
    try:
        return process.request({"op": "probe"})
    finally:
        process.offload()


class _WorkerProgress:
    """gr.Progress stand-in that forwards progress() calls to the parent."""

    def __init__(self, send) -> None:
        self._send = send

    def __call__(self, progress=None, desc=None, *args, **kwargs) -> None:
        self._send({"type": "progress", "value": progress, "desc": desc})


_ENGINES: dict = {}


def _whisper_engine(request: dict):
    key = ("whisper", tuple(sorted(request["dirs"].items())))
    if key not in _ENGINES:
        from modules.whisper.whisperx_inference import WhisperXInferenceEngine

        _ENGINES[key] = WhisperXInferenceEngine(**request["dirs"])
    return _ENGINES[key]


def _uvr_engine(request: dict):
    key = ("uvr", tuple(sorted(request["dirs"].items())))
    if key not in _ENGINES:
        from modules.uvr.music_separator import MusicSeparatorEngine

        _ENGINES[key] = MusicSeparatorEngine(**request["dirs"])
    return _ENGINES[key]


def _nllb_engine(request: dict):
    key = ("nllb", tuple(sorted(request["dirs"].items())))
    if key not in _ENGINES:
        from modules.translation.nllb_inference import NLLBInferenceEngine

        _ENGINES[key] = NLLBInferenceEngine(**request["dirs"])
    return _ENGINES[key]


def _probe() -> dict:
    from types import SimpleNamespace

    from modules.uvr.music_separator import UVR_AVAILABLE_MODELS, MusicSeparatorEngine
    from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline

    device = BaseTranscriptionPipeline.get_device()
    compute_types = BaseTranscriptionPipeline.get_available_compute_type(
        SimpleNamespace(device=device)
    )
    current_compute_type = BaseTranscriptionPipeline.get_compute_type(
        SimpleNamespace(device=device, available_compute_types=compute_types)
    )
    return {
        "device": device,
        "available_compute_types": compute_types,
        "current_compute_type": current_compute_type,
        "available_diarization_devices": BaseTranscriptionPipeline.available_diarization_devices.fget(
            SimpleNamespace(device=device)
        ),
        "uvr_device": MusicSeparatorEngine.get_device(),
        "uvr_available_models": UVR_AVAILABLE_MODELS,
    }


def _dispatch(request: dict, send) -> object:
    op = request.get("op")
    if op == "probe":
        return _probe()

    if op == "pipeline":
        engine = _whisper_engine(request)
        return engine._run_pipeline(
            request["audio"],
            _WorkerProgress(send),
            request["file_format"],
            request["add_timestamp"],
            None,
            *request["params"],
        )

    if op in ("separate", "separate_batch"):
        engine = _uvr_engine(request)
        audios = request["files"] if op == "separate_batch" else [
            request["audio"]]
        result = None
        for audio in audios:
            result = engine._separate(
                audio=audio,
                model_name=request["model_name"],
                device=request["device"],
                segment_size=request["segment_size"],
                save_file=request["save_file"],
                progress=_WorkerProgress(send),
            )
        return result[2] if op == "separate_batch" else result

    if op == "translate":
        engine = _nllb_engine(request)
        return engine.translate_file(progress=_WorkerProgress(send), **request["args"])

    raise WorkerError(f"Unknown inference op: {op!r}")


def child_main() -> None:
    # Real stdout is reserved for the pickled protocol; all logs go to stderr.
    sys.stdout = sys.stderr

    protocol_out = sys.__stdout__.buffer
    out_lock = threading.Lock()

    def send(message: dict) -> None:
        with out_lock:
            try:
                pickle.dump(message, protocol_out)
                protocol_out.flush()
            except OSError:
                # Parent is gone (pipe closed); exit so no orphan keeps VRAM.
                os._exit(1)

    while True:
        try:
            request = pickle.load(sys.stdin.buffer)
        except (EOFError, OSError, pickle.UnpicklingError, ValueError):
            return
        if request.get("op") == "shutdown":
            return
        try:
            payload = _dispatch(request, send)
        except BaseException as exc:
            send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            continue
        send({"type": "result", "payload": payload})


def _terminate_all() -> None:
    with _live_lock:
        processes = list(_live_processes)
    for process in processes:
        process.offload()


atexit.register(_terminate_all)

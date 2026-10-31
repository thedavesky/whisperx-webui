import importlib
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest

from modules.whisper.data_classes import (
    BGMSeparationParams,
    DiarizationParams,
    TranscriptionPipelineParams,
    WhisperParams,
)


class DummyMusicSeparator:
    available_models = ["uvr"]
    available_devices = ["cpu"]
    device = "cpu"

    @staticmethod
    def separate_files(*_, **__):
        return "instrumental", "vocals"


class DummyWhisperInference:
    def __init__(self, *_, **__):
        self.music_separator = DummyMusicSeparator()

    available_models = ["tiny"]
    available_langs = ["en"]
    available_compute_types = ["cpu"]
    current_compute_type = "cpu"
    device = "cpu"
    available_diarization_devices = ["cpu"]
    diarization_device = "cpu"
    available_diarization_compute_types = ["float32"]
    diarization_compute_type = "float32"

    @staticmethod
    def transcribe_file(*_, **__):
        return "ok", []

    @staticmethod
    def transcribe_youtube(*_, **__):
        return "ok", []

    @staticmethod
    def transcribe_mic(*_, **__):
        return "ok", []

    @staticmethod
    def set_cancel_event(*_, **__):
        return None


class DummyNllbInference:
    def __init__(self, *_, **__):
        pass

    available_models = ["small"]
    available_source_langs = ["en"]
    available_target_langs = ["fr"]

    @staticmethod
    def translate_file(*_, **__):
        return "ok", []


class DummyDeepLAPI:
    def __init__(self, *_, **__):
        pass

    available_source_langs = {"en": "English"}
    available_target_langs = {"fr": "French"}

    @staticmethod
    def translate_deepl(*_, **__):
        return "ok", []


FAKE_DEFAULTS = {
    "whisper": {
        "model_size": "tiny",
        "lang": "en",
        "file_format": "SRT",
        "is_translate": False,
        "add_timestamp": False,
    },
    "diarization": {
        "is_diarize": False,
        "diarization_device": "cpu",
        "compute_type": "float32",
        "hf_token": "",
        # Left over from an older release of the WebUI: the WhisperX diarization
        # toggle no longer exists and must not break launching or leak into params.
        "use_whisperx_diarization": True,
        "enable_offload": False,
        "assign_word_speakers": False,
        "fill_nearest_speaker": False,
    },
    "bgm_separation": {
        "is_separate_bgm": False,
        "uvr_model_size": "UVR-MDX-NET-Inst_HQ_4",
        "uvr_device": "cpu",
        "segment_size": 256,
        "save_file": False,
        "enable_offload": False,
    },
    "translation": {
        "add_timestamp": False,
        "deepl": {
            "api_key": "",
            "source_lang": "en",
            "target_lang": "fr",
            "is_pro": False,
        },
        "nllb": {
            "model_size": "small",
            "source_lang": "eng_Latn",
            "target_lang": "fra_Latn",
            "max_length": 10,
        },
    },
}


class DummyComponent:
    def __init__(self, *_, **kwargs):
        self.label = kwargs.get("label")
        self.value = kwargs.get("value")

    def click(self, *_, **__):
        return self

    def change(self, *_, **__):
        return self

    def stream(self, *_, **__):
        return self


class DummyContext(DummyComponent):
    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        return False


class DummyBlocks(DummyContext):
    launch_calls = {}

    def __init__(self, *, render=False, **kwargs):
        # Mirror a minimal subset of gr.Blocks kwargs so unexpected args (like deprecated css) raise
        super().__init__()
        allowed = {"theme", "delete_cache"}
        unexpected = set(kwargs) - allowed
        if unexpected:
            raise TypeError(f"DummyBlocks.__init__() got unexpected keyword arguments {unexpected}")
        self.render = render
        self.theme = kwargs.get("theme")
        self.delete_cache = kwargs.get("delete_cache")

    def queue(self, *_, **__):
        return self

    def launch(self, **__):
        DummyBlocks.launch_calls["called"] = True


def _build_gradio_stub(real_gr):
    """Layout containers are stubbed while components can be real or dummy."""
    return types.SimpleNamespace(
        Blocks=DummyBlocks,
        Row=DummyContext,
        Column=DummyContext,
        Accordion=DummyContext,
        TabItem=DummyContext,
        Tabs=DummyContext,
        Radio=DummyComponent,
        Files=DummyComponent,
        Textbox=DummyComponent,
        Dropdown=DummyComponent,
        Checkbox=DummyComponent,
        Button=DummyComponent,
        Microphone=DummyComponent,
        Number=DummyComponent,
        Slider=DummyComponent,
        Markdown=DummyComponent,
        Label=DummyComponent,
        Image=DummyComponent,
        HTML=DummyComponent,
        Audio=DummyComponent,
    )


def _build_component_gradio_stub(real_gr):
    """Same stub, but real components so labels/values can be inspected.

    Event handlers cannot be registered outside a real Blocks context, so this
    variant is only used for tests that stop before the UI is wired up.
    """
    stub = _build_gradio_stub(real_gr)
    for name in (
        "Radio",
        "Files",
        "Textbox",
        "Dropdown",
        "Checkbox",
        "Button",
        "Microphone",
        "Number",
        "Slider",
        "Markdown",
        "Label",
        "Image",
        "HTML",
        "Audio",
    ):
        setattr(stub, name, getattr(real_gr, name))
    return stub


def _setup_app(monkeypatch, tmp_path, gr_stub_factory):
    """Import app.py with every heavy dependency replaced by a stub."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    fake_whisperx_inference_module = types.ModuleType("modules.whisper.whisperx_inference")
    fake_whisperx_inference_module.WhisperXInference = DummyWhisperInference
    monkeypatch.setitem(sys.modules, "modules.whisper.whisperx_inference", fake_whisperx_inference_module)

    fake_nllb_module = types.ModuleType("modules.translation.nllb_inference")
    fake_nllb_module.NLLBInference = DummyNllbInference
    monkeypatch.setitem(sys.modules, "modules.translation.nllb_inference", fake_nllb_module)

    fake_deepl_module = types.ModuleType("modules.translation.deepl_api")
    fake_deepl_module.DeepLAPI = DummyDeepLAPI
    monkeypatch.setitem(sys.modules, "modules.translation.deepl_api", fake_deepl_module)

    fake_yt_module = types.ModuleType("modules.utils.youtube_manager")
    fake_yt_module.get_ytmetas = lambda *_, **__: (None, None, None)
    fake_yt_module.get_ytdata = lambda *_, **__: None
    fake_yt_module.get_ytaudio = lambda *_, **__: None
    monkeypatch.setitem(sys.modules, "modules.utils.youtube_manager", fake_yt_module)

    monkeypatch.setattr(sys, "argv", ["app.py"])
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "Translate", lambda *_, **__: nullcontext())

    import gradio as real_gr

    gr_stub = gr_stub_factory(real_gr)
    gr_stub.components = real_gr.components
    monkeypatch.setattr(app, "gr", gr_stub)

    def fake_load_yaml(path):
        if "i18n" in str(path).lower():
            return {"en": {"Language": "Language"}}
        return FAKE_DEFAULTS

    monkeypatch.setattr(app, "load_yaml", fake_load_yaml)
    DummyBlocks.launch_calls.clear()

    args = app.parser.parse_args([])
    args.output_dir = str(tmp_path)
    args.whisper_model_dir = str(tmp_path / "whisper")
    args.whisperx_model_dir = str(tmp_path / "whisperx")
    args.diarization_model_dir = str(tmp_path / "diarization")
    args.nllb_model_dir = str(tmp_path / "nllb")
    args.uvr_model_dir = str(tmp_path / "uvr")
    return app, args


@pytest.fixture
def app_module(monkeypatch, tmp_path):
    return _setup_app(monkeypatch, tmp_path, _build_gradio_stub)


@pytest.fixture
def app_with_components(monkeypatch, tmp_path):
    return _setup_app(monkeypatch, tmp_path, _build_component_gradio_stub)


def test_app_launch(app_module):
    app, args = app_module

    app_instance = app.App(args)
    app_instance.launch()

    assert DummyBlocks.launch_calls.get("called") is True


def test_pipeline_parameters_round_trip(app_with_components):
    """The Gradio components must map back onto the pipeline params positionally."""
    app, args = app_with_components
    app_instance = app.App(args)
    pipeline_inputs, _, _ = app_instance.create_pipeline_inputs()

    expected_count = (
        len(WhisperParams.model_fields)
        + len(DiarizationParams.model_fields)
        + len(BGMSeparationParams.model_fields)
    )
    assert len(pipeline_inputs) == expected_count
    assert len(app_instance.pipeline_param_defaults) == expected_count

    rebuilt = TranscriptionPipelineParams.from_list(list(app_instance.pipeline_param_defaults))
    assert rebuilt.whisper.model_size == FAKE_DEFAULTS["whisper"]["model_size"]
    assert rebuilt.whisper.batch_size == WhisperParams.model_fields["batch_size"].default
    assert rebuilt.diarization.diarization_device == "cpu"
    assert rebuilt.diarization.enable_offload is False
    assert rebuilt.bgm_separation.uvr_device == "cpu"
    assert rebuilt.bgm_separation.uvr_model_size == FAKE_DEFAULTS["bgm_separation"]["uvr_model_size"]


def test_diarization_provider_option_is_not_exposed(app_with_components):
    app, args = app_with_components
    app_instance = app.App(args)
    pipeline_inputs, _, _ = app_instance.create_pipeline_inputs()

    labels = [str(getattr(component, "label", "") or "") for component in pipeline_inputs]
    assert not any("whisperx diarization" in label.lower() for label in labels)
    assert "Diarization: Enable Diarization" in labels
    assert "Diarization: HuggingFace Token" in labels
    assert "Diarization: Tag Words With Speaker Labels" in labels
    assert "Diarization: Fallback to Nearest Speaker When No Overlap" in labels
    assert "Diarization: Offload sub model when finished" in labels

    assert "use_whisperx_diarization" not in app_instance.pipeline_param_names

import os
import argparse
import inspect
import threading
import gradio as gr
from gradio_i18n import Translate, gettext as _
import yaml
import re
import torch
import omegaconf

# --- PYTORCH 2.6+ PYANNOTE SECURITY FIX ---
# Pyannote weights use omegaconf.listconfig.ListConfig.
# We must explicitly tell PyTorch it is safe to load.
torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
# ------------------------------------------

from modules.utils.paths import (DIARIZATION_MODELS_DIR, OUTPUT_DIR, WHISPER_MODELS_DIR,
                                 NLLB_MODELS_DIR,
                                 DEFAULT_PARAMETERS_CONFIG_PATH, UVR_MODELS_DIR, I18N_YAML_PATH)
from modules.utils.files_manager import load_yaml, MEDIA_EXTENSION
from modules.whisper.whisperx_inference import WhisperXInference
from modules.translation.nllb_inference import NLLBInference
from modules.ui.htmls import *
from modules.utils.cli_manager import str2bool
from modules.utils.youtube_manager import get_ytmetas
from modules.translation.deepl_api import DeepLAPI
from modules.whisper.data_classes import *
from modules.utils.logger import get_logger


logger = get_logger()


class App:
    def __init__(self, args):
        self.args = args
        self.theme = self._resolve_theme(self.args.theme)
        # Check every 1 hour (3600) for cached files and delete them if older than 1 day (86400)
        self.app = self._build_blocks()
        self.whisper_inf = WhisperXInference(
            model_dir=self.args.whisperx_model_dir or self.args.whisper_model_dir,
            diarization_model_dir=self.args.diarization_model_dir,
            uvr_model_dir=self.args.uvr_model_dir,
            output_dir=self.args.output_dir,
        )
        self.cancel_event = threading.Event()
        self.whisper_inf.set_cancel_event(self.cancel_event)
        self.nllb_inf = NLLBInference(
            model_dir=self.args.nllb_model_dir,
            output_dir=os.path.join(self.args.output_dir, "translations")
        )
        self.deepl_api = DeepLAPI(
            output_dir=os.path.join(self.args.output_dir, "translations")
        )
        self.i18n = load_yaml(I18N_YAML_PATH)
        self.default_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        logger.info(f"Use \"{WhisperXInference.__name__}\" implementation\n"
                    f"Device \"{self.whisper_inf.device}\" is detected")

    @staticmethod
    def _blocks_signature_has(param_name: str) -> bool:
        """Return True if the current gr.Blocks accepts the kwarg."""
        try:
            signature = inspect.signature(gr.Blocks.__init__)
            return param_name in signature.parameters
        except (TypeError, ValueError):
            # Fallback if inspection fails
            return False

    def _build_blocks(self):
        blocks_kwargs = {}
        if self._blocks_signature_has("delete_cache"):
            blocks_kwargs["delete_cache"] = (3600, 86400)
        return gr.Blocks(**blocks_kwargs)

    @staticmethod
    def _component_accepts(component_cls, param_name: str) -> bool:
        """Return True if the provided Gradio component accepts the kwarg."""
        try:
            signature = inspect.signature(component_cls.__init__)
            return param_name in signature.parameters
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _pipeline_from_values(*pipeline_values):
        values = list(pipeline_values)
        whisper_split = len(WhisperParams.GRADIO_FIELDS)
        expected_count = (
            whisper_split
            + len(DiarizationParams.model_fields)
            + len(BGMSeparationParams.model_fields)
        )
        if len(values) != expected_count:
            raise ValueError(
                f"Expected {expected_count} pipeline parameters, but received {len(values)}. "
                "The rendered UI options and TranscriptionPipelineParams are out of sync."
            )
        whisper_params = WhisperParams(
            **dict(zip(WhisperParams.GRADIO_FIELDS, values[:whisper_split]))
        )
        return TranscriptionPipelineParams.from_list(
            whisper_params.to_list() + values[whisper_split:]
        )

    @staticmethod
    def _prefix_labels(components: list[gr.components.Component], prefix: str) -> None:
        """Prefix component labels to keep API parameter names human readable and unique."""
        for component in components:
            if hasattr(component, "label") and getattr(component, "label"):
                component.label = f"{prefix}: {component.label}"

    def _reset_cancellation(self):
        if self.cancel_event:
            self.cancel_event.clear()

    def _request_cancel(self):
        if self.cancel_event:
            self.cancel_event.set()
        return _("Cancellation requested. Stopping transcription...")

    def _transcribe_file(
        self,
        files,
        input_folder_path,
        include_subdirectory,
        save_same_dir,
        file_format,
        add_timestamp,
        *pipeline_values,
    ):
        self._reset_cancellation()
        params = self._pipeline_from_values(*pipeline_values)
        return self.whisper_inf.transcribe_file(
            files=files,
            input_folder_path=input_folder_path,
            include_subdirectory=include_subdirectory,
            save_same_dir=save_same_dir,
            file_format=file_format,
            add_timestamp=add_timestamp,
            pipeline_params=params,
        )

    def _transcribe_youtube(self, youtube_link, file_format, add_timestamp, *pipeline_values):
        self._reset_cancellation()
        params = self._pipeline_from_values(*pipeline_values)
        return self.whisper_inf.transcribe_youtube(
            youtube_link,
            file_format,
            add_timestamp,
            *params.to_list(),
        )

    def _transcribe_mic(self, mic_audio, file_format, add_timestamp, *pipeline_values):
        self._reset_cancellation()
        params = self._pipeline_from_values(*pipeline_values)
        return self.whisper_inf.transcribe_mic(
            mic_audio,
            file_format,
            add_timestamp,
            *params.to_list(),
        )

    @staticmethod
    def _resolve_theme(theme_arg):
        if not theme_arg:
            return None
        gr_themes = getattr(gr, "themes", None)
        if gr_themes and isinstance(theme_arg, str):
            theme_factory = getattr(gr_themes, theme_arg, None)
            if callable(theme_factory):
                try:
                    return theme_factory()
                except Exception:
                    logger.warning("Failed to build theme '%s', using raw value", theme_arg, exc_info=True)
        return theme_arg

    @staticmethod
    def _folder_option_updates(input_folder_path):
        enabled = bool((input_folder_path or "").strip())
        return gr.update(interactive=enabled), gr.update(interactive=enabled)

    @staticmethod
    def _pipeline_option_states(
        word_timestamps,
        is_diarize,
        is_separate_bgm,
        prefix,
    ) -> tuple:
        word_timestamps = bool(word_timestamps)
        is_diarize = bool(is_diarize)
        is_separate_bgm = bool(is_separate_bgm)
        return (
            is_separate_bgm,  # Stem separation model
            is_separate_bgm,  # Stem separation device
            is_separate_bgm,  # Stem separation segment size
            is_separate_bgm,  # Stem separation save files
            is_diarize,  # Diarization device
            is_diarize,  # Diarization compute type
            is_diarize,  # HuggingFace token
            is_diarize and word_timestamps,  # Word speaker labels
            is_diarize,  # Fallback to nearest speaker
            # Word timestamps come from WhisperX forced alignment,
            # which is where the word confidence scores come from.
            word_timestamps,  # Confidence scores
            not prefix,  # Hotwords have no effect while a prefix is set
        )

    def create_pipeline_inputs(self):
        whisper_params = self.default_params["whisper"]
        diarization_params = self.default_params["diarization"]
        uvr_params = self.default_params["bgm_separation"]

        with gr.Group():
            with gr.Row():
                dd_model = gr.Dropdown(
                    choices=self.whisper_inf.available_models,
                    value=whisper_params["model_size"],
                    label=_("Whisper Model"),
                    allow_custom_value=True,
                )
                dd_lang = gr.Dropdown(
                    choices=self.whisper_inf.available_langs + [AUTOMATIC_DETECTION],
                                      value=AUTOMATIC_DETECTION if whisper_params["lang"] == AUTOMATIC_DETECTION
                                      else whisper_params["lang"],
                    label=_("Whisper Language"),
                )
                dd_file_format = gr.Dropdown(
                    choices=["SRT", "WebVTT", "txt", "LRC"],
                    value=whisper_params["file_format"],
                    label=_("File Format"),
                )
            with gr.Row():
                cb_translate = gr.Checkbox(
                    value=whisper_params["is_translate"],
                    label=_("Translate to English? (Whisper)"),
                    interactive=True,
                )
            with gr.Row():
                cb_timestamp = gr.Checkbox(value=whisper_params["add_timestamp"],
                                           label=_("Add a timestamp to the end of the filename"),
                                           interactive=True)

        with gr.Accordion(_("Stem separation"), open=False, elem_classes="pipeline-options"):
            uvr_inputs = BGMSeparationParams.to_gradio_input(defaults=uvr_params,
                                                             available_models=self.whisper_inf.music_separator.available_models,
                                                             available_devices=self.whisper_inf.music_separator.available_devices,
                                                             device=self.whisper_inf.music_separator.device)

        with gr.Accordion(_("Diarization"), open=False, elem_classes="pipeline-options"):
            diarization_inputs = DiarizationParams.to_gradio_inputs(defaults=diarization_params,
                                                                    available_devices=self.whisper_inf.available_diarization_devices,
                                                                    device=self.whisper_inf.diarization_device,
                                                                    available_compute_types=self.whisper_inf.available_diarization_compute_types,
                                                                    compute_type=self.whisper_inf.diarization_compute_type)

        word_timing_accordion = gr.Accordion(_("Word timing"), open=False, elem_classes="pipeline-options")

        with gr.Accordion(_("Advanced parameters"), open=False, elem_classes="pipeline-options"):
            whisper_inputs = WhisperParams.to_gradio_inputs(defaults=whisper_params, only_advanced=True,
                                                            available_compute_types=self.whisper_inf.available_compute_types,
                                                            compute_type=self.whisper_inf.current_compute_type,
                                                            include_whisperx_controls=True)

        whisper_by_field = dict(zip(WhisperParams.GRADIO_FIELDS[3:], whisper_inputs))
        with word_timing_accordion:
            for field_name in ("word_timestamps", "show_confidence"):
                component = whisper_by_field[field_name]
                component.unrender()
                component.render()

        cb_separate_bgm, *uvr_dependents = uvr_inputs
        (
            cb_diarize,
            dd_diarization_device,
            dd_diarization_compute_type,
            tb_hf_token,
            cb_assign_word_speakers,
            cb_fill_nearest_speaker,
        ) = diarization_inputs
        option_state_inputs = [
            whisper_by_field["word_timestamps"],
            cb_diarize,
            cb_separate_bgm,
            whisper_by_field["prefix"],
        ]
        option_state_outputs = (
            list(uvr_dependents)
            + [
                dd_diarization_device,
                dd_diarization_compute_type,
                tb_hf_token,
                cb_assign_word_speakers,
                cb_fill_nearest_speaker,
            ]
            + [
                whisper_by_field["show_confidence"],
                whisper_by_field["hotwords"],
            ]
        )

        def _sync_option_states(*state_values):
            return [
                gr.update(interactive=enabled)
                for enabled in self._pipeline_option_states(*state_values)
            ]

        initial_states = self._pipeline_option_states(
            whisper_params.get("word_timestamps", False),
            diarization_params.get("is_diarize", False),
            uvr_params.get("is_separate_bgm", False),
            whisper_params.get("prefix") or "",
        )
        for initial_enabled, component in zip(initial_states, option_state_outputs):
            component.interactive = initial_enabled

        for component in option_state_inputs:
            component.change(
                fn=_sync_option_states,
                inputs=option_state_inputs,
                outputs=option_state_outputs,
                queue=False,
                api_name=False,
            )

        self._prefix_labels(whisper_inputs, _("Whisper"))
        self._prefix_labels(diarization_inputs, _("Diarization"))
        self._prefix_labels(uvr_inputs, _("BGM Separation"))

        pipeline_inputs = [dd_model, dd_lang, cb_translate] + whisper_inputs + diarization_inputs + uvr_inputs
        self.pipeline_param_names, self.pipeline_param_defaults = self._generate_param_names(pipeline_inputs)

        return (
            pipeline_inputs,
            dd_file_format,
            cb_timestamp
        )

    def _generate_param_names(self, components: list[gr.components.Component]) -> tuple[list[str], list]:
        """Create unique, snake_case parameter names and capture defaults from component values."""
        names: list[str] = []
        defaults: list = []
        seen: set[str] = set()
        for comp in components:
            raw_label = getattr(comp, "label", "") or "param"
            name = re.sub(r"[^0-9a-zA-Z]+", "_", str(raw_label)).strip("_").lower() or "param"
            candidate = name
            counter = 2
            while candidate in seen:
                candidate = f"{name}_{counter}"
                counter += 1
            seen.add(candidate)
            names.append(candidate)
            defaults.append(getattr(comp, "value", None))
        return names, defaults

    @staticmethod
    def _build_signature_params(names: list[str], defaults: list) -> list[inspect.Parameter]:
        params = []
        seen_default = False
        for name, default in zip(names, defaults):
            # If no default was set, keep the parameter required
            if default is not None and default is not inspect.Parameter.empty:
                param_default = default
                seen_default = True
            else:
                param_default = None if seen_default else inspect.Parameter.empty

            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=param_default,
                )
            )
        return params

    def _wrap_transcribe_file(self):
        """Provide a wrapper with explicit signature so Gradio names API params nicely."""
        base_param_names = [
            "files",
            "input_folder_path",
            "include_subdirectory",
            "save_same_dir",
            "file_format",
            "add_timestamp",
        ]
        base_defaults = [
            inspect.Parameter.empty,  # files required
            "",  # input_folder_path default
            False,
            True,
            "SRT",
            False,
        ]
        param_names = base_param_names + getattr(self, "pipeline_param_names", [])
        param_defaults = base_defaults + getattr(self, "pipeline_param_defaults", [])

        def handler(*args):
            files, input_folder_path, include_subdirectory, save_same_dir, file_format, add_timestamp, *pipeline_values = args
            return self._transcribe_file(
                files,
                input_folder_path,
                include_subdirectory,
                save_same_dir,
                file_format,
                add_timestamp,
                *pipeline_values,
            )

        handler.__signature__ = inspect.Signature(
            parameters=self._build_signature_params(param_names, param_defaults)
        )
        return handler

    def _wrap_transcribe_youtube(self):
        base_param_names = [
            "youtube_link",
            "file_format",
            "add_timestamp",
        ]
        base_defaults = [
            inspect.Parameter.empty,  # youtube_link required
            "SRT",
            False,
        ]
        param_names = base_param_names + getattr(self, "pipeline_param_names", [])
        param_defaults = base_defaults + getattr(self, "pipeline_param_defaults", [])

        def handler(*args):
            youtube_link, file_format, add_timestamp, *pipeline_values = args
            return self._transcribe_youtube(
                youtube_link,
                file_format,
                add_timestamp,
                *pipeline_values,
            )

        handler.__signature__ = inspect.Signature(
            parameters=self._build_signature_params(param_names, param_defaults)
        )
        return handler

    def _wrap_transcribe_mic(self):
        base_param_names = [
            "mic_audio",
            "file_format",
            "add_timestamp",
        ]
        base_defaults = [
            inspect.Parameter.empty,  # mic_audio required
            "SRT",
            False,
        ]
        param_names = base_param_names + getattr(self, "pipeline_param_names", [])
        param_defaults = base_defaults + getattr(self, "pipeline_param_defaults", [])

        def handler(*args):
            mic_audio, file_format, add_timestamp, *pipeline_values = args
            return self._transcribe_mic(
                mic_audio,
                file_format,
                add_timestamp,
                *pipeline_values,
            )

        handler.__signature__ = inspect.Signature(
            parameters=self._build_signature_params(param_names, param_defaults)
        )
        return handler

    def launch(self):
        translation_params = self.default_params["translation"]
        deepl_params = translation_params["deepl"]
        nllb_params = translation_params["nllb"]
        uvr_params = self.default_params["bgm_separation"]

        with self.app:
            transcription_events = []
            cancel_all_buttons = []
            gr.HTML(f"<style>{CSS}</style>")
            lang = gr.Radio(choices=list(self.i18n.keys()),
                            label=_("Language"), interactive=True,
                            visible=False,  # Set it by development purpose.
                            )
            with Translate(self.i18n):  # Add `lang = lang` here to test dynamic change of the languages.
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(MARKDOWN, elem_id="md_project")
                with gr.Tabs():
                    with gr.TabItem(_("File")):  # tab1
                        with gr.Column():
                            input_file = gr.Files(type="filepath", label=_("Upload File here"), file_types=MEDIA_EXTENSION)
                            tb_input_folder = gr.Textbox(label="Input Folder Path (Optional)",
                                                         info="Optional: Specify the folder path where the input files are located, if you prefer to use local files instead of uploading them."
                                                              " Leave this field empty if you do not wish to use a local path.",
                                                         visible=self.args.colab,
                                                         value="")
                            cb_include_subdirectory = gr.Checkbox(label="Include Subdirectory Files",
                                                                  info="When using Input Folder Path above, whether to include all files in the subdirectory or not.",
                                                                  visible=self.args.colab,
                                                                  value=False,
                                                                  interactive=False,
                                                                  elem_classes="folder-option")
                            cb_save_same_dir = gr.Checkbox(label="Save outputs at same directory",
                                                           info="When using Input Folder Path above, whether to save output in the same directory as inputs or not, in addition to the original"
                                                                " output directory.",
                                                           visible=self.args.colab,
                                                           value=True,
                                                           interactive=False,
                                                           elem_classes="folder-option")
                            tb_input_folder.change(
                                fn=self._folder_option_updates,
                                inputs=[tb_input_folder],
                                outputs=[cb_include_subdirectory, cb_save_same_dir],
                                queue=False,
                                api_name=False,
                            )
                        pipeline_params, dd_file_format, cb_timestamp = self.create_pipeline_inputs()

                        with gr.Row():
                            btn_run = gr.Button(_("GENERATE SUBTITLE FILE"), variant="primary")
                        btn_cancel_file = gr.Button(_("Cancel This Transcription"), variant="stop")
                        btn_cancel_all_file = gr.Button(_("Cancel All Transcriptions"), variant="secondary")
                        with gr.Row():
                            tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                            files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3, interactive=False)
                            btn_openfolder = gr.Button('📂', scale=1)

                        params = [input_file, tb_input_folder, cb_include_subdirectory, cb_save_same_dir,
                                  dd_file_format, cb_timestamp]
                        params = params + pipeline_params
                        file_run_event = btn_run.click(
                            fn=self._wrap_transcribe_file(),
                            inputs=params,
                            outputs=[tb_indicator, files_subtitles],
                            api_name="_transcribe_file",
                        )
                        transcription_events.append(file_run_event)
                        btn_cancel_file.click(
                            fn=self._request_cancel,
                            inputs=None,
                            outputs=[tb_indicator],
                            cancels=[file_run_event],
                        )
                        cancel_all_buttons.append((btn_cancel_all_file, tb_indicator))
                        btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)

                    with gr.TabItem(_("Youtube")):  # tab2
                        with gr.Row():
                            tb_youtubelink = gr.Textbox(label=_("Youtube Link"))
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                img_thumbnail = gr.Image(label=_("Youtube Thumbnail"))
                            with gr.Column():
                                tb_title = gr.Label(label=_("Youtube Title"))
                                tb_description = gr.Textbox(label=_("Youtube Description"), max_lines=15)

                        pipeline_params, dd_file_format, cb_timestamp = self.create_pipeline_inputs()

                        with gr.Row():
                            btn_run = gr.Button(_("GENERATE SUBTITLE FILE"), variant="primary")
                        btn_cancel_youtube = gr.Button(_("Cancel This Transcription"), variant="stop")
                        btn_cancel_all_youtube = gr.Button(_("Cancel All Transcriptions"), variant="secondary")
                        with gr.Row():
                            tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                            files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3)
                            btn_openfolder = gr.Button('📂', scale=1)

                        params = [tb_youtubelink, dd_file_format, cb_timestamp]

                        youtube_run_event = btn_run.click(
                            fn=self._wrap_transcribe_youtube(),
                            inputs=params + pipeline_params,
                            outputs=[tb_indicator, files_subtitles],
                            api_name="_transcribe_youtube",
                        )
                        transcription_events.append(youtube_run_event)
                        btn_cancel_youtube.click(
                            fn=self._request_cancel,
                            inputs=None,
                            outputs=[tb_indicator],
                            cancels=[youtube_run_event],
                        )
                        cancel_all_buttons.append((btn_cancel_all_youtube, tb_indicator))
                        tb_youtubelink.change(get_ytmetas, inputs=[tb_youtubelink],
                                              outputs=[img_thumbnail, tb_title, tb_description])
                        btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)

                    with gr.TabItem(_("Mic")):  # tab3
                        with gr.Row():
                            mic_kwargs = {
                                "label": _("Record with Mic"),
                                "type": "filepath",
                                "interactive": True,
                            }
                            if self._component_accepts(gr.Microphone, "show_download_button"):
                                mic_kwargs["show_download_button"] = True
                            mic_input = gr.Microphone(**mic_kwargs)

                        pipeline_params, dd_file_format, cb_timestamp = self.create_pipeline_inputs()

                        with gr.Row():
                            btn_run = gr.Button(_("GENERATE SUBTITLE FILE"), variant="primary")
                        btn_cancel_mic = gr.Button(_("Cancel This Transcription"), variant="stop")
                        btn_cancel_all_mic = gr.Button(_("Cancel All Transcriptions"), variant="secondary")
                        with gr.Row():
                            tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                            files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3)
                            btn_openfolder = gr.Button('📂', scale=1)

                        params = [mic_input, dd_file_format, cb_timestamp]

                        mic_run_event = btn_run.click(
                            fn=self._wrap_transcribe_mic(),
                            inputs=params + pipeline_params,
                            outputs=[tb_indicator, files_subtitles],
                            api_name="_transcribe_mic",
                        )
                        transcription_events.append(mic_run_event)
                        btn_cancel_mic.click(
                            fn=self._request_cancel,
                            inputs=None,
                            outputs=[tb_indicator],
                            cancels=[mic_run_event],
                        )
                        cancel_all_buttons.append((btn_cancel_all_mic, tb_indicator))
                        btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)

                    for cancel_button, indicator in cancel_all_buttons:
                        cancel_button.click(
                            fn=self._request_cancel,
                            inputs=None,
                            outputs=[indicator],
                            cancels=transcription_events,
                        )

                    with gr.TabItem(_("T2T Translation")):  # tab 4
                        with gr.Row():
                            file_subs = gr.Files(type="filepath", label=_("Upload Subtitle Files to translate here"))

                        with gr.TabItem(_("DeepL API")):  # sub tab1
                            with gr.Group():
                                with gr.Row():
                                    tb_api_key = gr.Textbox(label=_("Your Auth Key (API KEY)"),
                                                            value=deepl_params["api_key"])
                                with gr.Row():
                                    dd_source_lang = gr.Dropdown(label=_("Source Language"),
                                                                 value=AUTOMATIC_DETECTION if deepl_params["source_lang"] == AUTOMATIC_DETECTION
                                                                 else deepl_params["source_lang"],
                                                                 choices=list(self.deepl_api.available_source_langs.keys()))
                                    dd_target_lang = gr.Dropdown(label=_("Target Language"),
                                                                 value=deepl_params["target_lang"],
                                                                 choices=list(self.deepl_api.available_target_langs.keys()))
                                with gr.Row():
                                    cb_is_pro = gr.Checkbox(label=_("Pro User?"), value=deepl_params["is_pro"])
                                with gr.Row():
                                    cb_timestamp = gr.Checkbox(value=translation_params["add_timestamp"],
                                                               label=_("Add a timestamp to the end of the filename"),
                                                               interactive=True)
                            with gr.Row():
                                btn_run = gr.Button(_("TRANSLATE SUBTITLE FILE"), variant="primary")
                            with gr.Row():
                                tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                                files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3)
                                btn_openfolder = gr.Button('📂', scale=1)

                        btn_run.click(fn=self.deepl_api.translate_deepl,
                                      inputs=[tb_api_key, file_subs, dd_source_lang, dd_target_lang,
                                              cb_is_pro, cb_timestamp],
                                      outputs=[tb_indicator, files_subtitles],
                                      api_name="translate_deepl")

                        btn_openfolder.click(
                            fn=lambda: self.open_folder(os.path.join(self.args.output_dir, "translations")),
                            inputs=None,
                            outputs=None)

                        with gr.TabItem(_("NLLB")):  # sub tab2
                            with gr.Group():
                                with gr.Row():
                                    dd_model_size = gr.Dropdown(label=_("Model"), value=nllb_params["model_size"],
                                                                choices=self.nllb_inf.available_models)
                                    dd_source_lang = gr.Dropdown(label=_("Source Language"),
                                                                 value=nllb_params["source_lang"],
                                                                 choices=self.nllb_inf.available_source_langs)
                                    dd_target_lang = gr.Dropdown(label=_("Target Language"),
                                                                 value=nllb_params["target_lang"],
                                                                 choices=self.nllb_inf.available_target_langs)
                                with gr.Row():
                                    nb_max_length = gr.Number(label="Max Length Per Line", value=nllb_params["max_length"],
                                                              precision=0)
                                with gr.Row():
                                    cb_timestamp = gr.Checkbox(value=translation_params["add_timestamp"],
                                                               label=_("Add a timestamp to the end of the filename"),
                                                               interactive=True)
                            with gr.Row():
                                btn_run = gr.Button(_("TRANSLATE SUBTITLE FILE"), variant="primary")
                            with gr.Row():
                                tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                                files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3)
                                btn_openfolder = gr.Button('📂', scale=1)
                            with gr.Column():
                                md_vram_table = gr.HTML(NLLB_VRAM_TABLE, elem_id="md_nllb_vram_table")

                        btn_run.click(fn=self.nllb_inf.translate_file,
                                      inputs=[file_subs, dd_model_size, dd_source_lang, dd_target_lang,
                                              nb_max_length, cb_timestamp],
                                      outputs=[tb_indicator, files_subtitles],
                                      api_name="translate_nllb")

                        btn_openfolder.click(
                            fn=lambda: self.open_folder(os.path.join(self.args.output_dir, "translations")),
                            inputs=None,
                            outputs=None)

                    with gr.TabItem(_("BGM Separation")):
                        files_audio = gr.Files(type="filepath", label=_("Upload Audio Files to separate background music"))
                        with gr.Group():
                            dd_uvr_model_size = gr.Dropdown(label=_("Model"), value=uvr_params["uvr_model_size"],
                                                            choices=self.whisper_inf.music_separator.available_models)
                            dd_uvr_device = gr.Dropdown(label=_("Device"), value=self.whisper_inf.music_separator.device,
                                                        choices=self.whisper_inf.music_separator.available_devices)
                            nb_uvr_segment_size = gr.Number(label="Segment Size", value=uvr_params["segment_size"],
                                                            precision=0)
                        cb_uvr_save_file = gr.Checkbox(label=_("Save separated files to output"),
                                                       value=True, visible=False)
                        btn_run = gr.Button(_("SEPARATE BACKGROUND MUSIC"), variant="primary")
                        with gr.Column():
                            with gr.Row():
                                ad_instrumental = gr.Audio(label=_("Instrumental"), scale=8)
                                btn_open_instrumental_folder = gr.Button('📂', scale=1)
                            with gr.Row():
                                ad_vocals = gr.Audio(label=_("Vocals"), scale=8)
                                btn_open_vocals_folder = gr.Button('📂', scale=1)

                        btn_run.click(fn=self.whisper_inf.music_separator.separate_files,
                                      inputs=[files_audio, dd_uvr_model_size, dd_uvr_device, nb_uvr_segment_size,
                                              cb_uvr_save_file],
                                      outputs=[ad_instrumental, ad_vocals],
                                      api_name="separate_bgm")
                        btn_open_instrumental_folder.click(inputs=None,
                                                           outputs=None,
                                                           fn=lambda: self.open_folder(os.path.join(
                                                               self.args.output_dir, "UVR", "instrumental"
                                                           )))
                        btn_open_vocals_folder.click(inputs=None,
                                                     outputs=None,
                                                     fn=lambda: self.open_folder(os.path.join(
                                                         self.args.output_dir, "UVR", "vocals"
                                                     )))

        # Launch the app with optional gradio settings
        args = self.args
        self.app.queue(
            api_open=args.api_open
        ).launch(
            theme=self.theme,
            share=args.share,
            server_name=args.server_name,
            server_port=args.server_port,
            auth=(args.username, args.password) if args.username and args.password else None,
            root_path=args.root_path,
            inbrowser=args.inbrowser,
            ssl_verify=args.ssl_verify,
            ssl_keyfile=args.ssl_keyfile,
            ssl_keyfile_password=args.ssl_keyfile_password,
            ssl_certfile=args.ssl_certfile,
            allowed_paths=eval(args.allowed_paths) if args.allowed_paths else None
        )

    @staticmethod
    def open_folder(folder_path: str):
        if os.path.exists(folder_path):
            os.system(f"start {folder_path}")
        else:
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"The directory path {folder_path} has newly created.")


parser = argparse.ArgumentParser()
parser.add_argument('--share', type=str2bool, default=False, nargs='?', const=True, help='Gradio share value')
parser.add_argument('--server_name', type=str, default=None, help='Gradio server host')
parser.add_argument('--server_port', type=int, default=None, help='Gradio server port')
parser.add_argument('--root_path', type=str, default=None, help='Gradio root path')
parser.add_argument('--username', type=str, default=None, help='Gradio authentication username')
parser.add_argument('--password', type=str, default=None, help='Gradio authentication password')
parser.add_argument('--theme', type=str, default=None, help='Gradio Blocks theme')
parser.add_argument('--colab', type=str2bool, default=False, nargs='?', const=True, help='Is colab user or not')
parser.add_argument('--api_open', type=str2bool, default=True, nargs='?', const=True,
                    help='Enable api or not in Gradio')
parser.add_argument('--allowed_paths', type=str, default=None, help='Gradio allowed paths')
parser.add_argument('--inbrowser', type=str2bool, default=True, nargs='?', const=True,
                    help='Whether to automatically start Gradio app or not')
parser.add_argument('--ssl_verify', type=str2bool, default=True, nargs='?', const=True,
                    help='Whether to verify SSL or not')
parser.add_argument('--ssl_keyfile', type=str, default=None, help='SSL Key file location')
parser.add_argument('--ssl_keyfile_password', type=str, default=None, help='SSL Key file password')
parser.add_argument('--ssl_certfile', type=str, default=None, help='SSL cert file location')
parser.add_argument('--whisper_model_dir', type=str, default=WHISPER_MODELS_DIR,
                    help='Directory path of the Whisper model')
parser.add_argument('--whisperx_model_dir', type=str, default=None,
                    help='Directory path of the WhisperX model assets '
                         '(overrides --whisper_model_dir when provided)')
parser.add_argument('--diarization_model_dir', type=str, default=DIARIZATION_MODELS_DIR,
                    help='Directory path of the diarization model')
parser.add_argument('--nllb_model_dir', type=str, default=NLLB_MODELS_DIR,
                    help='Directory path of the Facebook NLLB model')
parser.add_argument('--uvr_model_dir', type=str, default=UVR_MODELS_DIR,
                    help='Directory path of the UVR model')
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory path of the outputs')
_args = parser.parse_args()

if __name__ == "__main__":
    app = App(args=_args)
    app.launch()

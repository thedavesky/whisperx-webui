import ast
import gradio as gr
import math
import torch
from typing import ClassVar, Optional, Dict, List, Union, NamedTuple
from fastapi import Query
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from gradio_i18n import Translate, gettext as _
from transformers.models.whisper.tokenization_whisper import LANGUAGES, TO_LANGUAGE_CODE
from copy import deepcopy
import yaml

from modules.utils.constants import *


class Segment(BaseModel):
    id: Optional[int] = Field(default=None, description="Incremental id for the segment")
    seek: Optional[int] = Field(default=None, description="Seek of the segment from chunked audio")
    text: Optional[str] = Field(default=None, description="Transcription text of the segment")
    start: Optional[float] = Field(default=None, description="Start time of the segment")
    end: Optional[float] = Field(default=None, description="End time of the segment")
    speaker: Optional[str] = Field(default=None, description="Speaker label for the segment")
    tokens: Optional[List[int]] = Field(default=None, description="List of token IDs")
    temperature: Optional[float] = Field(default=None, description="Temperature used during the decoding process")
    avg_logprob: Optional[float] = Field(default=None, description="Average log probability of the tokens")
    compression_ratio: Optional[float] = Field(default=None, description="Compression ratio of the segment")
    no_speech_prob: Optional[float] = Field(default=None, description="Probability that it's not speech")
    speaker: Optional[str] = Field(default=None, description="Speaker label for the segment")
    words: Optional[List['Word']] = Field(default=None, description="List of words contained in the segment")


class Word(BaseModel):
    start: Optional[float] = Field(default=None, description="Start time of the word")
    end: Optional[float] = Field(default=None, description="Start time of the word")
    word: Optional[str] = Field(default=None, description="Word text")
    probability: Optional[float] = Field(default=None, description="Probability of the word")
    speaker: Optional[str] = Field(default=None, description="Speaker label for the word")


class BaseParams(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), validate_assignment=True)

    def to_dict(self) -> Dict:
        return self.model_dump()

    def to_list(self) -> List:
        return list(self.model_dump().values())

    @classmethod
    def from_list(cls, data_list: List) -> 'BaseParams':
        field_names = list(cls.model_fields.keys())
        if len(data_list) != len(field_names):
            raise ValueError(
                f"{cls.__name__} requires exactly {len(field_names)} parameters, "
                f"received {len(data_list)}"
            )
        return cls(**dict(zip(field_names, data_list)))


# Models need to be wrapped with Field(Query()) to fix fastapi doc issue.
# More info : https://github.com/fastapi/fastapi/discussions/8634#discussioncomment-5153136
class DiarizationParams(BaseParams):
    """Speaker diarization parameters"""
    is_diarize: bool = Field(default=False, description="Enable speaker diarization")
    diarization_device: str = Field(default="cuda", min_length=1, description="Device to run Diarization model.")
    diarization_compute_type: str = Field(
        default="float32",
        min_length=1,
        description="Computation type for diarization (separate from the ASR compute_type)"
    )
    hf_token: str = Field(
        default="",
        description="Hugging Face token for downloading diarization models"
    )
    assign_word_speakers: bool = Field(
        default=False,
        description="Attach speaker labels to aligned words"
    )
    fill_nearest_speaker: bool = Field(
        default=False,
        description="When no overlap exists, tag the nearest diarization speaker"
    )

    @classmethod
    def to_gradio_inputs(cls,
                         defaults: Optional[Dict] = None,
                         available_devices: Optional[List] = None,
                         device: Optional[str] = None,
                         available_compute_types: Optional[List] = None,
                         diarization_compute_type: Optional[str] = None) -> List[gr.components.base.FormComponent]:
        defaults = defaults or {}
        device_choices = ["cpu", "cuda", "xpu"] if available_devices is None else available_devices
        default_device = defaults.get("diarization_device", device)
        if device_choices and default_device not in device_choices:
            default_device = device_choices[0]
        compute_type_choices = available_compute_types or [cls.model_fields["diarization_compute_type"].default]
        compute_type_value = defaults.get("diarization_compute_type", diarization_compute_type)
        if compute_type_value not in compute_type_choices:
            compute_type_value = compute_type_choices[0]
        return [
            gr.Checkbox(
                label=_("Enable Diarization"),
                value=defaults.get("is_diarize", cls.model_fields["is_diarize"].default),
            ),
            gr.Dropdown(
                label=_("Device"),
                choices=device_choices,
                value=default_device,
            ),
            gr.Dropdown(
                label=_("Compute Type"),
                choices=compute_type_choices,
                value=compute_type_value,
                interactive=True,
            ),
            gr.Textbox(
                label=_("HuggingFace Token"),
                value=defaults.get("hf_token", cls.model_fields["hf_token"].default),
                info=_("This is only needed the first time you download the model")
            ),
            gr.Checkbox(
                label=_("Tag Words With Speaker Labels"),
                value=defaults.get("assign_word_speakers", cls.model_fields["assign_word_speakers"].default),
                info=_("Add diarization speaker tags to each aligned word"),
            ),
            gr.Checkbox(
                label=_("Fallback to Nearest Speaker When No Overlap"),
                value=defaults.get("fill_nearest_speaker", cls.model_fields["fill_nearest_speaker"].default),
                info=_("Use the closest diarization segment when a word has no overlap"),
            )
        ]


class BGMSeparationParams(BaseParams):
    """Background music separation parameters"""
    is_separate_bgm: bool = Field(default=False, description="Enable background music separation")
    uvr_model_size: str = Field(
        default="UVR-MDX-NET-Inst_HQ_4",
        min_length=1,
        description="UVR model size"
    )
    uvr_device: str = Field(default="cuda", min_length=1, description="Device to run UVR model.")
    segment_size: int = Field(
        default=256,
        gt=0,
        description="Segment size for UVR model"
    )
    save_file: bool = Field(
        default=False,
        description="Whether to save separated audio files"
    )

    @classmethod
    def to_gradio_input(cls,
                        defaults: Optional[Dict] = None,
                        available_devices: Optional[List] = None,
                        device: Optional[str] = None,
                        available_models: Optional[List] = None) -> List[gr.components.base.FormComponent]:
        defaults = defaults or {}
        device_choices = ["cpu", "cuda", "xpu"] if available_devices is None else available_devices
        default_device = defaults.get("uvr_device", device)
        if device_choices and default_device not in device_choices:
            default_device = device_choices[0]
        return [
            gr.Checkbox(
                label=_("Enable Background Music Remover Filter"),
                value=defaults.get("is_separate_bgm", cls.model_fields["is_separate_bgm"].default),
                interactive=True,
                info=_("Enabling this will remove background music")
            ),
            gr.Dropdown(
                label=_("Model"),
                choices=["UVR-MDX-NET-Inst_HQ_4",
                         "UVR-MDX-NET-Inst_3"] if available_models is None else available_models,
                value=defaults.get("uvr_model_size", cls.model_fields["uvr_model_size"].default),
            ),
            gr.Dropdown(
                label=_("Device"),
                choices=device_choices,
                value=default_device,
            ),
            gr.Number(
                label="Segment Size",
                value=defaults.get("segment_size", cls.model_fields["segment_size"].default),
                precision=0,
                info="Segment size for UVR model"
            ),
            gr.Checkbox(
                label=_("Save separated files to output"),
                value=defaults.get("save_file", cls.model_fields["save_file"].default),
            )
        ]


class WhisperParams(BaseParams):
    """Whisper parameters"""
    model_size: str = Field(default="large-v2", min_length=1, description="Whisper model size")
    lang: Optional[str] = Field(default=None, description="Source language of the file to transcribe")
    is_translate: bool = Field(default=False, description="Translate speech to English end-to-end")
    beam_size: int = Field(default=5, ge=1, description="Beam size for decoding")
    log_prob_threshold: float = Field(
        default=-1.0,
        description="Threshold for average log probability of sampled tokens"
    )
    no_speech_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for detecting silence"
    )
    compute_type: str = Field(default="int8", min_length=1, description="Computation type for transcription")
    best_of: int = Field(default=5, ge=1, description="Number of candidates when sampling")
    patience: float = Field(default=1.0, gt=0, description="Beam search patience factor")
    condition_on_previous_text: bool = Field(
        default=True,
        description="Use previous output as prompt for next window"
    )
    prompt_reset_on_temperature: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Temperature threshold for resetting prompt"
    )
    initial_prompt: Optional[str] = Field(default=None, description="Initial prompt for first window")
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Temperature for sampling"
    )
    compression_ratio_threshold: float = Field(
        default=2.4,
        gt=0,
        description="Threshold for gzip compression ratio"
    )
    length_penalty: float = Field(default=1.0, gt=0, description="Exponential length penalty")
    repetition_penalty: float = Field(default=1.0, gt=0, description="Penalty for repeated tokens")
    no_repeat_ngram_size: int = Field(default=0, ge=0, description="Size of n-grams to prevent repetition")
    prefix: Optional[str] = Field(default=None, description="Prefix text for first window")
    suppress_blank: bool = Field(
        default=True,
        description="Suppress blank outputs at start of sampling"
    )
    suppress_tokens: Optional[Union[List[int], str]] = Field(default=[-1], description="Token IDs to suppress")
    max_initial_timestamp: float = Field(
        default=1.0,
        ge=0.0,
        description="Maximum initial timestamp"
    )
    word_timestamps: bool = Field(
        default=False,
        description="Extract word-level timestamps with WhisperX forced alignment"
    )
    prepend_punctuations: Optional[str] = Field(
        default="\"'“¿([{-",
        description="Punctuations to merge with next word"
    )
    append_punctuations: Optional[str] = Field(
        default="\"'.。,，!！?？:：”)]}、",
        description="Punctuations to merge with previous word"
    )
    max_new_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum number of new tokens per chunk")
    chunk_length: Optional[int] = Field(default=30, gt=0, description="Length of audio segments in seconds")
    hallucination_silence_threshold: Optional[float] = Field(
        default=None,
        description="Threshold for skipping silent periods in hallucination detection"
    )
    hotwords: Optional[str] = Field(default=None, description="Hotwords/hint phrases for the model")
    language_detection_threshold: Optional[float] = Field(
        default=0.5,
        description="Threshold for language detection probability (not used by the WhisperX pipeline; keep the default)",
    )
    language_detection_segments: int = Field(
        default=1,
        gt=0,
        description="Number of segments for language detection (not used by the WhisperX pipeline; keep the default)",
    )
    batch_size: int = Field(default=24, gt=0, description="Batch size for processing")
    whisperx_confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum WhisperX confidence score to keep aligned words"
    )
    whisperx_vad_onset: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Pyannote VAD onset threshold used by WhisperX to start speech chunks"
    )
    whisperx_vad_offset: float = Field(
        default=0.363,
        gt=0.0,
        lt=1.0,
        description="Pyannote VAD offset threshold used by WhisperX to end speech chunks"
    )
    show_confidence: bool = Field(
        default=False,
        description="Display confidence scores for segments and words",
    )

    @field_validator('lang')
    def validate_lang(cls, v):
        """Accept a language code, its display name, or the automatic-detection label."""
        if v in (None, AUTOMATIC_DETECTION):
            return None
        if v in LANGUAGES:
            return v
        if v in TO_LANGUAGE_CODE:
            return TO_LANGUAGE_CODE[v]
        raise ValueError(
            f"Unsupported language '{v}'. Use a Whisper language code, a display name, "
            f"or '{AUTOMATIC_DETECTION}'."
        )

    @field_validator('suppress_tokens', mode="before")
    def validate_supress_tokens(cls, v):
        if v is None:
            return [-1]
        if isinstance(v, str):
            # One query-string value can carry the whole list as a literal, e.g. "[-1]".
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Invalid Suppress Tokens. The value must be type of List[int]: {e}")
        if not isinstance(v, list):
            if isinstance(v, int) and not isinstance(v, bool):
                # A single `?suppress_tokens=-1` param is the one-token list.
                return [v]
            raise ValueError("Invalid Suppress Tokens. The value must be type of List[int]")
        tokens = []
        for token in v:
            if isinstance(token, str):
                # Repeated `?suppress_tokens=1&suppress_tokens=2` params arrive as strings.
                try:
                    token = ast.literal_eval(token)
                except (ValueError, SyntaxError):
                    raise ValueError("Invalid Suppress Tokens. The value must be type of List[int]")
            if isinstance(token, bool) or not isinstance(token, int):
                raise ValueError("Invalid Suppress Tokens. The value must be type of List[int]")
            tokens.append(token)
        return tokens

    @model_validator(mode="after")
    def validate_finite_numbers(self):
        for name, value in self.model_dump().items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        return self

    GRADIO_FIELDS: ClassVar[List[str]] = [
        "model_size",
        "lang",
        "is_translate",
        "beam_size",
        "compute_type",
        "patience",
        "initial_prompt",
        "length_penalty",
        "prefix",
        "suppress_blank",
        "suppress_tokens",
        "word_timestamps",
        "chunk_length",
        "hotwords",
        "batch_size",
        "whisperx_vad_onset",
        "whisperx_vad_offset",
        "show_confidence",
    ]

    @classmethod
    def to_gradio_inputs(
        cls,
        defaults: Optional[Dict] = None,
        only_advanced: Optional[bool] = True,
        available_models: Optional[List] = None,
        available_langs: Optional[List] = None,
        available_compute_types: Optional[List] = None,
        compute_type: Optional[str] = None,
        include_whisperx_controls: bool = True,
    ):
        defaults = defaults or {}
        inputs = []
        if not only_advanced:
            inputs += [
                gr.Dropdown(
                    label=_("Model"),
                    choices=available_models,
                    value=defaults.get("model_size", cls.model_fields["model_size"].default),
                ),
                gr.Dropdown(
                    label=_("Language"),
                    choices=available_langs,
                    value=defaults.get("lang", AUTOMATIC_DETECTION),
                ),
                gr.Checkbox(
                    label=_("Translate to English?"),
                    value=defaults.get("is_translate", cls.model_fields["is_translate"].default),
                ),
            ]

        compute_type_value = defaults.get("compute_type", compute_type)
        if (
            available_compute_types
            and compute_type_value not in available_compute_types
        ):
            compute_type_value = compute_type or available_compute_types[0]

        compute_type_dropdown = gr.Dropdown(
            label="Compute Type",
            choices=["float16", "int8", "int16"] if available_compute_types is None else available_compute_types,
            value=compute_type_value,
            info="Computation type for transcription"
        )
        beam_size_number = gr.Number(
            label="Beam Size",
            value=defaults.get("beam_size", cls.model_fields["beam_size"].default),
            precision=0,
            info="Beam size for decoding"
        )

        inputs += [
            beam_size_number,
            compute_type_dropdown,
            gr.Number(
                label="Patience",
                value=defaults.get("patience", cls.model_fields["patience"].default),
                info="Beam search patience factor"
            ),
            gr.Textbox(
                label="Initial Prompt",
                value=defaults.get("initial_prompt", GRADIO_NONE_STR),
                info="Initial prompt for first window"
            ),
            gr.Number(
                label="Length Penalty",
                value=defaults.get("length_penalty", cls.model_fields["length_penalty"].default),
                info="Exponential length penalty"
            ),
            gr.Textbox(
                label="Prefix",
                value=defaults.get("prefix", GRADIO_NONE_STR),
                info="Prefix text for first window"
            ),
            gr.Checkbox(
                label="Suppress Blank",
                value=defaults.get("suppress_blank", cls.model_fields["suppress_blank"].default),
                info="Suppress blank outputs at start of sampling"
            ),
            gr.Textbox(
                label="Suppress Tokens",
                value=defaults.get("suppress_tokens", "[-1]"),
                info="Token IDs to suppress"
            ),
            gr.Checkbox(
                label="Word Timestamps",
                value=defaults.get("word_timestamps", cls.model_fields["word_timestamps"].default),
                info="Extract word-level timestamps using WhisperX forced alignment"
            ),
        ]

        if include_whisperx_controls:
            inputs += [
                gr.Number(
                    label="Chunk Length (s)",
                    value=defaults.get("chunk_length", cls.model_fields["chunk_length"].default),
                    precision=0,
                    info="Length of audio segments in seconds",
                ),
                gr.Textbox(
                    label="Hotwords",
                    value=defaults.get("hotwords", GRADIO_NONE_STR),
                    info="Hotwords/hint phrases for the model",
                ),
                gr.Number(
                    label="Batch Size",
                    value=defaults.get("batch_size", cls.model_fields["batch_size"].default),
                    precision=0,
                    info="Batch size for processing",
                ),
                gr.Slider(
                    minimum=0.01,
                    maximum=0.99,
                    step=0.001,
                    label=_("Pyannote VAD Onset Threshold"),
                    value=defaults.get(
                        "whisperx_vad_onset",
                        cls.model_fields["whisperx_vad_onset"].default,
                    ),
                    info=_("WhisperX's built-in speech filter. Scores above this value start a speech chunk"),
                ),
                gr.Slider(
                    minimum=0.01,
                    maximum=0.99,
                    step=0.001,
                    label=_("Pyannote VAD Offset Threshold"),
                    value=defaults.get(
                        "whisperx_vad_offset",
                        cls.model_fields["whisperx_vad_offset"].default,
                    ),
                    info=_("WhisperX's built-in speech filter. Scores below this value end a speech chunk"),
                ),
                gr.Checkbox(
                    label=_("Show Confidence Scores"),
                    value=defaults.get(
                        "show_confidence",
                        cls.model_fields["show_confidence"].default,
                    ),
                    info=_("Append confidence values to segment and word outputs"),
                ),
            ]

        return inputs


class TranscriptionPipelineParams(BaseModel):
    """Transcription pipeline parameters"""
    whisper: WhisperParams = Field(default_factory=WhisperParams)
    diarization: DiarizationParams = Field(default_factory=DiarizationParams)
    bgm_separation: BGMSeparationParams = Field(default_factory=BGMSeparationParams)

    def to_dict(self) -> Dict:
        data = {
            "whisper": self.whisper.to_dict(),
            "diarization": self.diarization.to_dict(),
            "bgm_separation": self.bgm_separation.to_dict()
        }
        return data

    def to_list(self) -> List:
        """
        Convert data class to the list because I have to pass the parameters as a list in the gradio.
        Related Gradio issue: https://github.com/gradio-app/gradio/issues/2471
        See more about Gradio pre-processing: https://www.gradio.app/docs/components
        """
        whisper_list = self.whisper.to_list()
        diarization_list = self.diarization.to_list()
        bgm_sep_list = self.bgm_separation.to_list()
        return whisper_list + diarization_list + bgm_sep_list

    @staticmethod
    def from_list(pipeline_list: List) -> 'TranscriptionPipelineParams':
        """Convert list to the data class again to use it in a function."""
        data_list = deepcopy(pipeline_list)

        whisper_expected = len(WhisperParams.model_fields)
        diarization_expected = len(DiarizationParams.model_fields)
        bgm_expected = len(BGMSeparationParams.model_fields)
        expected = whisper_expected + diarization_expected + bgm_expected
        if len(data_list) != expected:
            raise ValueError(
                f"TranscriptionPipelineParams requires exactly {expected} parameters, "
                f"received {len(data_list)}"
            )

        return TranscriptionPipelineParams(
            whisper=WhisperParams.from_list(data_list[:whisper_expected]),
            diarization=DiarizationParams.from_list(
                data_list[whisper_expected:whisper_expected + diarization_expected]
            ),
            bgm_separation=BGMSeparationParams.from_list(data_list[whisper_expected + diarization_expected:])
        )

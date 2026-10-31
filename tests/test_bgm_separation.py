import gradio as gr
import pytest
import torch
import os

from modules.utils.paths import *
from modules.whisper.data_classes import *
from test_config import *
from test_transcription import download_file, run_asr_pipeline


@pytest.mark.skipif(
    not is_cuda_available(),
    reason="Skipping because the test only works on GPU"
)
def test_bgm_separation_pipeline():
    run_asr_pipeline(bgm_separation=True, diarization=False)

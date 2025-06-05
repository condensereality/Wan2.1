import pytest

from wan.configs import WAN_CONFIGS
from wan.model_loader import VaceWanModelLoader

@pytest.fixture
def original_1_3_B_checkpoint():
    return "/home/ollie/projects/layerhaus/src/py/condense-video2video/models/vace-1.3B"

@pytest.fixture
def original_14B_checkpoint():
    return "/home/ollie/projects/layerhaus/src/py/condense-video2video/models/vace-14B"

@pytest.fixture
def fp8_14B_checkpoint():
    return "/home/ollie/projects/layerhaus/src/py/condense-video2video/models/vace-14B-fp8"

def test_model_loader_bfloat16_1_3B(original_1_3_B_checkpoint):
    loader = VaceWanModelLoader()
    type = "vace-1.3B"
    loader.load_model(original_1_3_B_checkpoint, type)

def test_model_loader_fp8_14B(fp8_14B_checkpoint):
    loader = VaceWanModelLoader()
    type = "vace-14B"
    loader.load_model(fp8_14B_checkpoint, type)

def test_model_loader_bfloat16_14B(original_14B_checkpoint):
    loader = VaceWanModelLoader()
    type = "vace-14B"
    loader.load_model(original_14B_checkpoint, type)

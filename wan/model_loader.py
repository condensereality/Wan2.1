import json
import logging
import os
from pathlib import Path
from typing import Optional, NamedTuple, List, Dict

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from easydict import EasyDict
from safetensors.torch import load_file
from tqdm import tqdm

from wan.modules import VaceWanModel

DEFAULT_1_3B_CONFIG : EasyDict = EasyDict({
    "dim": 1536,
    "eps": 1e-06,
    "ffn_dim": 8960,
    "freq_dim": 256,
    "in_dim": 16,
    "model_type": "vace",
    "num_heads": 12,
    "num_layers": 30,
    "out_dim": 16,
    "text_len": 512,
    "vace_layers": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
    "vace_in_dim": 96
})

DEFAULT_14B_CONFIG : EasyDict = EasyDict({
    "dim": 5120,
    "eps": 1e-06,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "in_dim": 16,
    "model_type": "vace",
    "num_heads": 40,
    "num_layers": 40,
    "out_dim": 16,
    "text_len": 512,
    "vace_layers": [0, 5, 10, 15, 20, 25, 30, 35],
    "vace_in_dim": 96})

DEFAULT_CONFIGS : EasyDict = EasyDict({
    "vace-1.3B" : DEFAULT_1_3B_CONFIG,
    "vace-14B" : DEFAULT_14B_CONFIG,
})

def get_model_files(checkpoint_dir : str | os.PathLike) -> List[Path]:
    """
    Fetches and returns a list of all model files in a directory with the `.safetensors`
    extension. If no such files are found, an exception is raised.

    This function is specifically designed to filter and collect files that use the
    `.safetensors` extension. It iterates through the provided directory, gathers
    relevant files, and raises an error if none match the expected criteria.

    Args:
        checkpoint_dir (str | os.PathLike): The path to the directory where the function
            will search for `.safetensors` files. The directory should exist and be
            accessible for this operation.

    Returns:
        List[Path]: A list of pathlib `Path` objects representing the `.safetensors`
        files found in the specified directory.

    Raises:
        FileNotFoundError: If no files with the `.safetensors` extension are found
        in the specified directory.
    """
    files = [x for x in Path(checkpoint_dir).iterdir()]
    output = []
    for file in files:
        if file.suffix == ".safetensors":
            output.append(file)

    if len(output) == 0:
        raise FileNotFoundError(f"Could not find any safe tensor model files in {checkpoint_dir}")
    return output

def get_model_config(
        model_type : str,
        checkpoint_dir : str | os.PathLike) -> EasyDict:
    # if there is a config file in the checkpoint dir, load it and use it
    config_path = Path(checkpoint_dir) / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = EasyDict(json.load(f))
        return config

    logging.info(f"no config file found at {config_path}. using default config")
    return DEFAULT_CONFIGS[model_type]


def load_state_dict(model_files : List[Path]) -> Dict[str, torch.Tensor]:
    output = {}
    for model_file in model_files:
        sd  = load_file(model_file)
        output.update(sd)
    return output


def validate_weights(state_dict : Dict[str, torch.Tensor],
                     model_type : str):
    if "vace_blocks.0.after_proj.weight" not in state_dict:
        raise ValueError(f"missing vace blocks in model weights")
    dim = state_dict["patch_embedding.weight"].shape[0]

    expected_dim = DEFAULT_CONFIGS[model_type].dim
    if dim  != expected_dim:
        raise ValueError(f"{model_type} dim should be {expected_dim}. got {dim} instead")

def get_quantisation(state_dict : Dict[str, torch.Tensor]) -> torch.dtype:
    for k, v in state_dict.items():
        if v.dtype == torch.float8_e4m3fn:
            return v.dtype
    return torch.bfloat16



def set_model_weights(
        model : VaceWanModel,
        state_dict : Dict[str, torch.Tensor],
        base_type : torch.dtype,
        quantisation : torch.dtype):
    params_to_keep = {"norm", "head", "bias", "time_in", "vector_in", "patch_embedding", "time_", "img_emb", "modulation", "text_embedding", "adapter"}

    param_count = sum(1 for _ in model.named_parameters())
    for name, param in tqdm(model.named_parameters(),
                            total=param_count,
                            leave=True,
                            desc=f"setting weights to model"):
        keep_block = any(p in name for p in params_to_keep)
        dtype_to_use = base_type if keep_block else quantisation
        if "patch_embedding" in name:
            dtype_to_use = torch.float32
        set_module_tensor_to_device(model, name, device="cpu", dtype=dtype_to_use, value=state_dict[name])



class VaceWanModelLoader:
    def __init__(self):
        pass

    def load_model(self,
                   checkpoint_dir : str | os.PathLike,
                   model_type: str,
                   vace_module_path : Optional[str | os.PathLike] = None,
                   lora_path : Optional[str | os.PathLike] = None) -> VaceWanModel:

        if vace_module_path is not None:
            raise NotImplementedError("separate vace modules not supported")
        if lora_path is not None:
            raise NotImplementedError("lora modules not supported")

        logging.info(f"loading model type {model_type} from {checkpoint_dir}")

        if model_type not in ["vace-1.3B", "vace-14B"]:
            raise NotImplementedError(f"model_type {model_type} is not supported")

        # get the model config
        model_config = get_model_config(model_type, checkpoint_dir)

        # initialise the model based on the config
        logging.info("initialising model")
        with init_empty_weights():
            model = VaceWanModel(**model_config)
        model.eval()

        model_files = get_model_files(checkpoint_dir)
        state_dict = load_state_dict(model_files)
        validate_weights(state_dict, model_type)

        quantisation = get_quantisation(state_dict)
        set_model_weights(model, state_dict, base_type=torch.bfloat16, quantisation=quantisation)

        return model

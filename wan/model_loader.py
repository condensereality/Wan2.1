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

# Supported model types
SUPPORTED_MODEL_TYPES = ["vace-1.3B", "vace-14B"]

# Default configuration for the 1.3B parameter model
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

# Default configuration for the 14B parameter model
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
    "vace_in_dim": 96
})

# Mapping of model types to their default configurations
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
    # Get all .safetensors files in the directory
    output = list(Path(checkpoint_dir).glob("*.safetensors"))

    if not output:
        raise FileNotFoundError(f"Could not find any safe tensor model files in {checkpoint_dir}")
    return output

def get_model_config(
        model_type : str,
        checkpoint_dir : str | os.PathLike) -> EasyDict:
    """
    Retrieves the model configuration either from a config file in the checkpoint
    directory or falls back to the default configuration for the specified model type.

    The function first checks if a 'config.json' file exists in the checkpoint directory.
    If it does, the configuration is loaded from that file. Otherwise, it uses the
    predefined default configuration for the specified model type.

    Args:
        model_type (str): The type of model to load (e.g., "vace-1.3B", "vace-14B").
            Must be one of the supported model types.
        checkpoint_dir (str | os.PathLike): The directory path where the model checkpoint
            and potentially a config file are stored.

    Returns:
        EasyDict: A dictionary-like object containing the model configuration parameters.

    Raises:
        KeyError: If the specified model_type is not found in the DEFAULT_CONFIGS.
    """
    # Check if a config file exists in the checkpoint directory
    config_path = Path(checkpoint_dir) / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = EasyDict(json.load(f))
        return config

    # Fall back to default configuration if no config file is found
    logging.info(f"No config file found at {config_path}. Using default config for {model_type}.")
    return DEFAULT_CONFIGS[model_type]


def load_state_dict(model_files : List[Path]) -> Dict[str, torch.Tensor]:
    """
    Loads and combines model weights from multiple safetensors files into a single state dictionary.

    This function iterates through a list of safetensors files, loads each one, and combines
    them into a single state dictionary. This is useful when model weights are split across
    multiple files.

    Args:
        model_files (List[Path]): A list of paths to safetensors files containing model weights.

    Returns:
        Dict[str, torch.Tensor]: A combined state dictionary containing all model weights.

    Note:
        If the same parameter key exists in multiple files, the later files will
        overwrite the values from earlier files.
    """
    output = {}
    for model_file in model_files:
        sd = load_file(model_file)
        output.update(sd)
    return output


def validate_weights(state_dict : Dict[str, torch.Tensor],
                     model_type : str) -> None:
    """
    Validates that the loaded model weights are compatible with the specified model type.

    This function performs two key validations:
    1. Checks that the state dictionary contains the expected VACE blocks
    2. Verifies that the model dimension matches the expected dimension for the model type

    Args:
        state_dict (Dict[str, torch.Tensor]): The model state dictionary containing weights
        model_type (str): The type of model being loaded (e.g., "vace-1.3B", "vace-14B")

    Raises:
        ValueError: If the state dictionary is missing expected keys or if the model
            dimensions don't match the expected dimensions for the specified model type
        KeyError: If the model_type is not found in DEFAULT_CONFIGS or if required keys
            are missing in the state dictionary
    """
    # Check for required keys in the state dictionary
    if "vace_blocks.0.after_proj.weight" not in state_dict:
        raise ValueError(f"Missing VACE blocks in model weights. The state dictionary appears to be incomplete or corrupted.")

    if "patch_embedding.weight" not in state_dict:
        raise KeyError(f"Missing 'patch_embedding.weight' in state dictionary. Cannot validate model dimensions.")

    # Validate model dimensions
    dim = state_dict["patch_embedding.weight"].shape[0]
    expected_dim = DEFAULT_CONFIGS[model_type].dim

    if dim != expected_dim:
        raise ValueError(f"Model dimension mismatch for {model_type}. Expected: {expected_dim}, got: {dim}. "
                         f"The weights may be for a different model type.")

def get_quantisation(state_dict : Dict[str, torch.Tensor]) -> torch.dtype:
    """
    Determines the quantization data type to use for the model weights.

    This function examines the state dictionary to check if any tensors are using
    the float8_e4m3fn data type (a quantized format). If found, it returns this data type
    to maintain compatibility with the quantized weights. Otherwise, it defaults to bfloat16.

    Args:
        state_dict (Dict[str, torch.Tensor]): The model state dictionary containing weights

    Returns:
        torch.dtype: The data type to use for model quantization. Returns torch.float8_e4m3fn
            if any tensor in the state dictionary uses this type, otherwise returns torch.bfloat16.
    """
    # Check if any tensor uses float8_e4m3fn quantization
    for _, tensor in state_dict.items():
        if tensor.dtype == torch.float8_e4m3fn:
            return tensor.dtype

    # Default to bfloat16 if no float8 quantization is found
    return torch.bfloat16



def set_model_weights(
        model : VaceWanModel,
        state_dict : Dict[str, torch.Tensor],
        base_type : torch.dtype,
        quantisation : torch.dtype) -> None:
    """
    Sets the weights of the model from the state dictionary with appropriate data types.

    This function applies the weights from the state dictionary to the model, using different
    data types for different parameters based on their importance and computational requirements.
    Critical parameters (like normalization layers, heads, biases) use the base_type, while
    other parameters use the quantisation type to reduce memory usage. Patch embedding
    parameters specifically use float32 for maximum precision.

    Args:
        model (VaceWanModel): The model to which weights will be applied
        state_dict (Dict[str, torch.Tensor]): Dictionary containing model weights
        base_type (torch.dtype): Data type to use for critical parameters (typically torch.bfloat16)
        quantisation (torch.dtype): Data type to use for non-critical parameters (may be quantized)

    Returns:
        None

    Note:
        All weights are set on the CPU device. If GPU usage is required, the model should be
        moved to the appropriate device after this function completes.
    """
    # Parameters that should use the base_type rather than quantisation
    params_to_keep = {
        "norm", "head", "bias", "time_in", "vector_in", "patch_embedding", 
        "time_", "img_emb", "modulation", "text_embedding", "adapter"
    }

    # Count total parameters for progress bar
    param_count = sum(1 for _ in model.named_parameters())

    # Set each parameter with the appropriate data type
    for name, _ in tqdm(model.named_parameters(),
                        total=param_count,
                        leave=True,
                        desc="Setting weights to model"):
        # Determine which data type to use based on parameter name
        keep_block = any(p in name for p in params_to_keep)
        dtype_to_use = base_type if keep_block else quantisation

        # Patch embedding always uses float32 for maximum precision
        if "patch_embedding" in name:
            dtype_to_use = torch.float32

        # Set the parameter value with the appropriate data type
        set_module_tensor_to_device(model, name, device="cpu", dtype=dtype_to_use, value=state_dict[name])



class VaceWanModelLoader:
    """
    A class for loading VaceWan models from checkpoint directories.

    This class provides functionality to load VaceWan models with different parameter sizes
    (1.3B or 14B) from checkpoint directories. It handles loading the model configuration,
    initializing the model with empty weights, loading the state dictionary from safetensors
    files, validating the weights, and setting the weights with appropriate data types.
    """

    def load_model(self,
                   checkpoint_dir : str | os.PathLike,
                   model_type: str,
                   vace_module_path : Optional[str | os.PathLike] = None,
                   lora_path : Optional[str | os.PathLike] = None) -> VaceWanModel:
        """
        Loads a VaceWan model from a checkpoint directory.

        This method performs the following steps:
        1. Validates the model type and unsupported features
        2. Loads the model configuration
        3. Initializes the model with empty weights
        4. Loads the state dictionary from safetensors files
        5. Validates the weights against the model type
        6. Sets the weights with appropriate data types

        Args:
            checkpoint_dir (str | os.PathLike): Directory containing the model checkpoint files
            model_type (str): Type of model to load (e.g., "vace-1.3B", "vace-14B")
            vace_module_path (Optional[str | os.PathLike]): Path to separate VACE modules
                (not currently supported)
            lora_path (Optional[str | os.PathLike]): Path to LoRA modules
                (not currently supported)

        Returns:
            VaceWanModel: The loaded model in evaluation mode

        Raises:
            NotImplementedError: If unsupported features are requested or if the model type
                is not supported
            ValueError, KeyError: If the model weights validation fails
        """
        # Check for unsupported features
        if vace_module_path is not None:
            raise NotImplementedError("Separate VACE modules are not supported")
        if lora_path is not None:
            raise NotImplementedError("LoRA modules are not supported")

        logging.info(f"Loading model type {model_type} from {checkpoint_dir}")

        # Validate model type
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise NotImplementedError(f"Model type '{model_type}' is not supported. "
                                     f"Supported types are: {', '.join(SUPPORTED_MODEL_TYPES)}")

        # Get the model configuration
        model_config = get_model_config(model_type, checkpoint_dir)

        # Initialize the model with empty weights
        logging.info("Initializing model")
        with init_empty_weights():
            model = VaceWanModel(**model_config)
        model.eval()

        # Load and validate the model weights
        model_files = get_model_files(checkpoint_dir)
        state_dict = load_state_dict(model_files)
        validate_weights(state_dict, model_type)

        # Set the model weights with appropriate data types
        quantisation = get_quantisation(state_dict)
        set_model_weights(model, state_dict, base_type=torch.bfloat16, quantisation=quantisation)

        return model

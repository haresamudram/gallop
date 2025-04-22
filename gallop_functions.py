import yaml, argparse, os
import torch
from typing import Type
import logging

NoneType = Type[None]

LOGGER = logging.getLogger(os.getenv("LOGGING_NAME", "GalLoP"))

def read_yaml_to_cfg_gallop(yaml_file_path):
    """
    Load arguments from a YAML file and use them to setup the cfg object.

    Args:
        yaml_file_path: Path to the YAML file.

    Returns:
        cfg: Config object initialized with the YAML file contents.
    """
    with open(yaml_file_path, "r") as f:
        args_dict = yaml.safe_load(f)

    # Convert dictionary to Namespace for `setup_cfg`
    args_namespace = argparse.Namespace(**args_dict)
    
    return args_namespace

def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> NoneType:
    if (checkpoint_path is not None) and (checkpoint_path.lower() != "none"):
        LOGGER.info(f"Loading checkpoint {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cuda")
        keys = model.load_trainable_state_dict(state_dict["state_dict"], strict=False)
        if len(keys.unexpected_keys) > 0:
            raise RuntimeError(f"Unexpected keys in state_dict: {keys.unexpected_keys}")
        if len(keys.missing_keys) > 0:
            LOGGER.warning(f"Missing keys in state_dict: {keys.missing_keys}")
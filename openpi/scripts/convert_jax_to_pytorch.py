"""Convert JAX checkpoint to PyTorch safetensors format."""
import argparse
import json
import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_jax_to_pytorch(jax_checkpoint_dir: str, output_dir: str):
    """Convert JAX checkpoint to PyTorch safetensors format."""
    from openpi.models import model as _model
    from openpi.training import config as _config
    
    jax_dir = Path(jax_checkpoint_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load JAX params
    logger.info(f"Loading JAX checkpoint from {jax_dir}")
    params = _model.restore_params(jax_dir / "params", dtype=jnp.bfloat16)
    
    # Flatten the nested dict
    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    flat_params = flatten_dict(params)
    
    # Convert to PyTorch tensors
    pytorch_state_dict = {}
    for key, value in flat_params.items():
        # Convert JAX array to numpy, then to PyTorch
        np_array = np.array(value)
        # Handle bfloat16 by converting to float32 first, then to bfloat16 in torch
        if np_array.dtype.name == 'bfloat16':
            # Convert bfloat16 to float32 for numpy compatibility
            np_array = np_array.astype(np.float32)
            pytorch_state_dict[key] = torch.from_numpy(np_array).to(torch.bfloat16)
        else:
            pytorch_state_dict[key] = torch.from_numpy(np_array)
        
    # Save as safetensors
    output_path = out_dir / "model.safetensors"
    logger.info(f"Saving PyTorch checkpoint to {output_path}")
    save_file(pytorch_state_dict, str(output_path))
    
    # Copy config if exists
    config_src = jax_dir / "config.json"
    if config_src.exists():
        import shutil
        shutil.copy(config_src, out_dir / "config.json")
    else:
        # Create a default config
        config = {
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m", 
            "action_dim": 32,
            "action_horizon": 50,
            "tokenizer_max_length": 200
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    # Copy assets
    assets_src = jax_dir / "assets"
    if assets_src.exists():
        import shutil
        shutil.copytree(assets_src, out_dir / "assets", dirs_exist_ok=True)
    
    logger.info("Conversion complete!")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JAX checkpoint directory")
    parser.add_argument("--output", required=True, help="Output directory for PyTorch checkpoint")
    args = parser.parse_args()
    
    convert_jax_to_pytorch(args.input, args.output)

"""
Integration with ComfyUI's standard model loading system.

This module registers TurboDiffusion models with ComfyUI's model management
so they can be used with standard nodes like DiffusionModelLoader.
"""

import folder_paths
import torch
from pathlib import Path


def register_turbodiffusion_models():
    """
    Register TurboDiffusion model paths with ComfyUI's folder_paths system.

    This allows TurboDiffusion models to be discovered by standard ComfyUI nodes.
    """
    # Add TurboDiffusion models to the diffusion_models search paths
    # This is already done automatically since we're using the diffusion_models folder

    # Get the current diffusion_models paths
    diffusion_paths = folder_paths.get_folder_paths("diffusion_models")
    print(f"TurboDiffusion: Using model paths: {diffusion_paths}")


def load_turbodiffusion_checkpoint(checkpoint_path: str):
    """
    Load a TurboDiffusion checkpoint in a format compatible with ComfyUI.

    Args:
        checkpoint_path: Path to the TurboDiffusion .pth file

    Returns:
        Loaded model checkpoint
    """
    print(f"Loading TurboDiffusion checkpoint: {checkpoint_path}")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    return checkpoint


def is_turbodiffusion_model(checkpoint_path: str) -> bool:
    """
    Check if a checkpoint file is a TurboDiffusion model.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        True if this is a TurboDiffusion model
    """
    path = Path(checkpoint_path)

    # Check filename patterns
    turbodiffusion_patterns = [
        "TurboWan",
        "turbodiffusion",
        "turbo-diffusion",
    ]

    filename = path.name.lower()
    return any(pattern.lower() in filename for pattern in turbodiffusion_patterns)


def get_turbodiffusion_vae_path(model_path: str) -> str:
    """
    Get the VAE path for a TurboDiffusion model.

    Args:
        model_path: Path to main TurboDiffusion model

    Returns:
        Path to VAE checkpoint
    """
    model_dir = Path(model_path).parent
    vae_path = model_dir / "Wan2.1_VAE.pth"

    if not vae_path.exists():
        # Try other locations
        for search_dir in folder_paths.get_folder_paths("diffusion_models"):
            alt_vae = Path(search_dir) / "Wan2.1_VAE.pth"
            if alt_vae.exists():
                return str(alt_vae)

    return str(vae_path)


def get_turbodiffusion_t5_path(model_path: str) -> str:
    """
    Get the T5 encoder path for a TurboDiffusion model.

    Args:
        model_path: Path to main TurboDiffusion model

    Returns:
        Path to T5 encoder checkpoint
    """
    model_dir = Path(model_path).parent
    t5_path = model_dir / "models_t5_umt5-xxl-enc-bf16.pth"

    if not t5_path.exists():
        # Try other locations
        for search_dir in folder_paths.get_folder_paths("diffusion_models"):
            alt_t5 = Path(search_dir) / "models_t5_umt5-xxl-enc-bf16.pth"
            if alt_t5.exists():
                return str(alt_t5)

    return str(t5_path)


# Register on module import
try:
    register_turbodiffusion_models()
except Exception as e:
    print(f"Warning: Could not register TurboDiffusion models: {e}")

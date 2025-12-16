"""TurboDiffusion VAE Loader node for ComfyUI."""

from typing import Tuple
from pathlib import Path

from ..utils.model_management import get_checkpoint_dir


class TurboWanVAELoader:
    """
    ComfyUI node for loading Wan2.1 VAE encoder/decoder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the node."""
        return {
            "required": {
                "vae_name": (
                    cls._get_vae_list(),
                    {
                        "default": "Wan2.1_VAE.pth",
                        "tooltip": "Select the VAE checkpoint to load",
                    },
                ),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "video/turbodiffusion"
    DESCRIPTION = "Load Wan2.1 VAE for TurboDiffusion video encoding/decoding"

    @classmethod
    def _get_vae_list(cls):
        """Get list of available VAE checkpoints."""
        vae_files = []

        # Search in diffusion_models folder
        try:
            import folder_paths
            for search_dir in folder_paths.get_folder_paths("diffusion_models"):
                search_path = Path(search_dir)
                if search_path.exists():
                    for vae_file in search_path.glob("*VAE*.pth"):
                        vae_files.append(vae_file.name)
        except:
            pass

        # Search in local checkpoints
        checkpoint_dir = get_checkpoint_dir()
        if checkpoint_dir.exists():
            for vae_file in checkpoint_dir.glob("*VAE*.pth"):
                if vae_file.name not in vae_files:
                    vae_files.append(vae_file.name)

        # Default if none found
        if not vae_files:
            vae_files = ["Wan2.1_VAE.pth"]

        return sorted(vae_files)

    def load_vae(self, vae_name: str) -> Tuple[dict]:
        """
        Load the VAE checkpoint.

        Args:
            vae_name: Name of VAE checkpoint file

        Returns:
            Tuple containing VAE configuration dictionary
        """
        print(f"\n{'='*60}")
        print(f"Loading TurboWan VAE")
        print(f"{'='*60}")
        print(f"VAE: {vae_name}")

        # Find VAE file
        vae_path = None

        # Search in diffusion_models folder
        try:
            import folder_paths
            for search_dir in folder_paths.get_folder_paths("diffusion_models"):
                search_path = Path(search_dir) / vae_name
                if search_path.exists():
                    vae_path = search_path
                    break
        except:
            pass

        # Search in local checkpoints
        if vae_path is None:
            checkpoint_dir = get_checkpoint_dir()
            local_vae = checkpoint_dir / vae_name
            if local_vae.exists():
                vae_path = local_vae

        if vae_path is None:
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"ERROR: VAE not found: {vae_name}\n"
                f"{'='*60}\n"
                f"Please download from:\n"
                f"https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B\n"
                f"\nPlace in: ComfyUI/models/diffusion_models/\n"
                f"{'='*60}\n"
            )

        vae_config = {
            "vae_path": str(vae_path),
            "vae_name": vae_name,
        }

        print(f"VAE loaded from: {vae_path}")
        print(f"{'='*60}\n")

        return (vae_config,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Cache based on VAE name."""
        return kwargs.get("vae_name", "")

    @classmethod
    def VALIDATE_INPUTS(cls, vae_name):
        """Validate input parameters."""
        if not isinstance(vae_name, str) or not vae_name:
            return "vae_name must be a non-empty string"
        return True

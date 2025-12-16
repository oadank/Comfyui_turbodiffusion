"""TurboWan2.2 Model Loader node for ComfyUI."""

from typing import Tuple, Optional
import folder_paths

from ..utils.model_management import (
    clear_cuda_cache,
)


class TurboWan2ModelLoader:
    """
    ComfyUI node for loading TurboWan2.2 I2V models.

    Works like the standard Load Diffusion Model node - pick any .pth file.
    Supports both high noise and low noise variants.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the node."""
        return {
            "required": {
                "model_filename": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {
                        "tooltip": "Select TurboWan model file (.pth). Works with both high and low noise variants.",
                    },
                ),
            },
            "optional": {
                "vae": (
                    "VAE",
                    {
                        "tooltip": "VAE from Load TurboWan VAE node. If not connected, will auto-load Wan2.1_VAE.pth.",
                    },
                ),
                "text_encoder": (
                    "TEXT_ENCODER",
                    {
                        "tooltip": "T5 encoder from Load TurboWan T5 Encoder node. If not connected, will auto-load.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("TURBODIFFUSION_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "video/turbodiffusion"
    DESCRIPTION = "Load TurboWan2.2 I2V model (high or low noise variant)"

    def load_model(
        self,
        model_filename: str,
        vae: Optional[dict] = None,
        text_encoder: Optional[dict] = None,
    ) -> Tuple[dict]:
        """
        Load the TurboDiffusion model.

        Args:
            model_filename: Model filename from diffusion_models folder
            vae: Optional VAE dict from TurboWanVAELoader
            text_encoder: Optional text encoder dict from TurboWanT5Loader

        Returns:
            Tuple containing model configuration dictionary
        """
        from pathlib import Path

        # Clear CUDA cache before loading
        clear_cuda_cache()

        print(f"\n{'='*60}")
        print(f"Loading TurboWan2.2 Model")
        print(f"{'='*60}")
        print(f"Model file: {model_filename}")

        try:
            # Get full path to model file
            model_path = folder_paths.get_full_path("diffusion_models", model_filename)
            if not model_path:
                raise FileNotFoundError(f"Model file not found: {model_filename}")

            model_path = Path(model_path)
            print(f"Model path: {model_path}")

            # Auto-load VAE if not provided
            if vae is None:
                print("VAE not provided, auto-loading Wan2.1_VAE.pth...")
                vae_paths = folder_paths.get_folder_paths("diffusion_models")
                vae_path = None
                for search_dir in vae_paths:
                    vae_file = Path(search_dir) / "Wan2.1_VAE.pth"
                    if vae_file.exists():
                        vae_path = str(vae_file)
                        break

                if not vae_path:
                    raise FileNotFoundError(
                        "Wan2.1_VAE.pth not found. Please use TurboWan VAE Loader node "
                        "or place Wan2.1_VAE.pth in ComfyUI/models/diffusion_models/"
                    )

                vae = {
                    "vae_path": vae_path,
                    "vae_name": "Wan2.1_VAE.pth"
                }
                print(f"Auto-loaded VAE: {vae_path}")
            else:
                print(f"Using provided VAE: {vae.get('vae_name', 'unknown')}")

            # Auto-load T5 encoder if not provided
            if text_encoder is None:
                print("T5 encoder not provided, auto-loading...")
                t5_paths = folder_paths.get_folder_paths("diffusion_models")
                t5_path = None
                for search_dir in t5_paths:
                    t5_file = Path(search_dir) / "models_t5_umt5-xxl-enc-bf16.pth"
                    if t5_file.exists():
                        t5_path = str(t5_file)
                        break

                if not t5_path:
                    raise FileNotFoundError(
                        "models_t5_umt5-xxl-enc-bf16.pth not found. Please use TurboWan T5 Encoder Loader node "
                        "or place models_t5_umt5-xxl-enc-bf16.pth in ComfyUI/models/diffusion_models/"
                    )

                text_encoder = {
                    "t5_path": t5_path,
                    "encoder_name": "models_t5_umt5-xxl-enc-bf16.pth"
                }
                print(f"Auto-loaded T5 encoder: {t5_path}")
            else:
                print(f"Using provided T5 encoder: {text_encoder.get('encoder_name', 'unknown')}")

            # Build model configuration
            model_config = {
                "model_path": str(model_path),
                "model_name": model_filename,
                "vae_path": vae["vae_path"],
                "vae_name": vae.get("vae_name", "unknown"),
                "t5_path": text_encoder["t5_path"],
                "t5_name": text_encoder.get("encoder_name", "unknown"),
                "device": "cuda" if self._is_cuda_available() else "cpu",
            }

            print(f"{'='*60}")
            print(f"Model loaded successfully!")
            print(f"Model: {model_config['model_name']}")
            print(f"VAE: {model_config['vae_name']}")
            print(f"T5: {model_config['t5_name']}")
            print(f"{'='*60}\n")

            return (model_config,)

        except FileNotFoundError as e:
            error_msg = (
                f"\n{'='*60}\n"
                f"ERROR: Model files not found!\n"
                f"{'='*60}\n"
                f"{str(e)}\n\n"
                f"Required files:\n"
                f"1. TurboWan model: TurboWan2.2-I2V-A14B-*.pth\n"
                f"2. VAE: Wan2.1_VAE.pth\n"
                f"3. T5 Encoder: models_t5_umt5-xxl-enc-bf16.pth\n\n"
                f"Download from: https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P\n"
                f"Place in: ComfyUI/models/diffusion_models/\n"
                f"{'='*60}\n"
            )
            print(error_msg)
            raise RuntimeError(error_msg) from e

        except Exception as e:
            error_msg = (
                f"\n{'='*60}\n"
                f"ERROR: Failed to load model!\n"
                f"{'='*60}\n"
                f"{str(e)}\n"
                f"{'='*60}\n"
            )
            print(error_msg)
            raise RuntimeError(error_msg) from e

    @staticmethod
    def _is_cuda_available() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Determine if the node needs to be re-executed.

        This method tells ComfyUI when to re-execute the node.
        Return a different value each time to force re-execution,
        or return a consistent value to cache results.
        """
        # Cache based on model filename
        return kwargs.get('model_filename', '')

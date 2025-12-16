"""TurboDiffusion I2V Generator node for ComfyUI."""

from typing import Tuple
import torch

from ..utils.preprocessing import comfyui_to_pil, video_to_comfyui
from ..utils.model_management import clear_cuda_cache


class TurboDiffusionI2V:
    """
    ComfyUI node for TurboDiffusion Image-to-Video generation.

    This node takes an input image and generates a video sequence using
    the TurboDiffusion I2V model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the node."""
        return {
            "required": {
                "model": (
                    "TURBODIFFUSION_MODEL",
                    {
                        "tooltip": "TurboDiffusion model from Model Loader node",
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Input image to animate (ComfyUI IMAGE format)",
                    },
                ),
                "num_steps": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                        "tooltip": "Number of sampling steps. More steps = higher quality but slower. "
                                   "TurboDiffusion is optimized for 1-4 steps.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Random seed for reproducible generation",
                    },
                ),
            },
            "optional": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional text prompt to guide video generation",
                    },
                ),
                "num_frames": (
                    "INT",
                    {
                        "default": 77,
                        "min": 9,
                        "max": 121,
                        "step": 4,
                        "tooltip": "Number of frames to generate. More frames = longer video but more VRAM.",
                    },
                ),
                "sigma_max": (
                    "FLOAT",
                    {
                        "default": 200.0,
                        "min": 0.0,
                        "max": 500.0,
                        "step": 0.1,
                        "tooltip": "Initial noise level for rCM sampling. Higher = more variation.",
                    },
                ),
                "boundary": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Noise schedule switching point for dual-expert model. "
                                   "Controls transition between high/low noise experts.",
                    },
                ),
                "attention_type": (
                    ["original", "sla", "sagesla"],
                    {
                        "default": "original",
                        "tooltip": "Attention mechanism. SLA/SageSLA are faster but may reduce quality. "
                                   "SageSLA requires spargeattn package.",
                    },
                ),
                "sla_topk": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Sparse attention ratio for SLA/SageSLA. Lower = faster but less detail.",
                    },
                ),
                "use_ode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use ODE-based sampling for sharper outputs. May increase generation time.",
                    },
                ),
                "adaptive_resolution": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Adapt to input image resolution. Disable to force model's native resolution.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "frame_count")
    FUNCTION = "generate_video"
    CATEGORY = "video/turbodiffusion"
    DESCRIPTION = "Generate video from image using TurboDiffusion I2V model"

    def generate_video(
        self,
        model: dict,
        image: torch.Tensor,
        num_steps: int,
        seed: int,
        prompt: str = "",
        num_frames: int = 77,
        sigma_max: float = 200.0,
        boundary: float = 0.9,
        attention_type: str = "original",
        sla_topk: float = 0.1,
        use_ode: bool = False,
        adaptive_resolution: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate video from input image.

        Args:
            model: Model configuration from TurboWan2ModelLoader
            image: Input image tensor [B, H, W, 3] (ComfyUI format)
            num_steps: Number of sampling steps (1-4)
            seed: Random seed
            prompt: Optional text prompt
            num_frames: Number of frames to generate
            sigma_max: Initial noise level
            boundary: Noise schedule switching point
            attention_type: Attention mechanism type
            sla_topk: Sparse attention ratio
            use_ode: Whether to use ODE sampling
            adaptive_resolution: Whether to adapt to input resolution

        Returns:
            Tuple of (frame_batch [N,H,W,3], frame_count)
        """
        print(f"\n{'='*60}")
        print(f"TurboDiffusion I2V Generation")
        print(f"{'='*60}")
        print(f"Input image shape: {image.shape}")
        print(f"Num steps: {num_steps}")
        print(f"Num frames: {num_frames}")
        print(f"Seed: {seed}")
        print(f"Prompt: {prompt if prompt else '(none)'}")
        print(f"Sigma max: {sigma_max}")
        print(f"Boundary: {boundary}")
        print(f"Attention type: {attention_type}")
        print(f"{'='*60}\n")

        try:
            # Set random seed
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            # Convert ComfyUI IMAGE to PIL Image
            input_image = comfyui_to_pil(image)
            print(f"Converted to PIL Image: {input_image.size}")

            # Import turbodiffusion (lazy import to avoid errors if not installed)
            try:
                import turbodiffusion
                print(f"TurboDiffusion version: {turbodiffusion.__version__ if hasattr(turbodiffusion, '__version__') else 'unknown'}")
            except ImportError as e:
                raise ImportError(
                    "TurboDiffusion package not found. "
                    "Please install it with: pip install turbodiffusion --no-build-isolation\n"
                    "Or install all dependencies with: uv sync"
                ) from e

            # Check for SageSLA if requested
            if attention_type == "sagesla":
                try:
                    import spargeattn
                except ImportError:
                    print(
                        "WARNING: SageSLA attention requested but spargeattn not installed. "
                        "Falling back to 'sla' attention.\n"
                        "Install with: pip install git+https://github.com/thu-ml/SpargeAttn.git"
                    )
                    attention_type = "sla"

            # TODO: Actual TurboDiffusion inference implementation
            # This is a placeholder that will need to be replaced with actual turbodiffusion API calls
            # The exact API depends on how the turbodiffusion package is structured

            print("=" * 60)
            print("NOTE: This is a placeholder implementation!")
            print("=" * 60)
            print("The actual TurboDiffusion inference needs to be integrated once")
            print("the turbodiffusion package API is properly documented.")
            print("=" * 60)
            print("\nFor now, creating a dummy video output...")
            print("To complete this implementation, you'll need to:")
            print("1. Load the actual model using the turbodiffusion package")
            print("2. Preprocess the input image according to model requirements")
            print("3. Run the inference pipeline with the specified parameters")
            print("4. Post-process the output frames")
            print("=" * 60 + "\n")

            # Placeholder: Create dummy frames (copy input image multiple times)
            # In real implementation, this would be replaced with actual video generation
            dummy_frames = []
            for i in range(num_frames):
                # This would be replaced with actual generated frames
                dummy_frames.append(input_image)

            # Convert to ComfyUI format
            output_frames = video_to_comfyui(dummy_frames)

            print(f"Generated {num_frames} frames")
            print(f"Output shape: {output_frames.shape}")
            print(f"{'='*60}\n")

            # Clear CUDA cache
            clear_cuda_cache()

            return (output_frames, num_frames)

        except ImportError as e:
            error_msg = (
                f"\n{'='*60}\n"
                f"ERROR: Missing dependency!\n"
                f"{'='*60}\n"
                f"{str(e)}\n"
                f"{'='*60}\n"
            )
            print(error_msg)
            raise RuntimeError(error_msg) from e

        except torch.cuda.OutOfMemoryError as e:
            error_msg = (
                f"\n{'='*60}\n"
                f"ERROR: CUDA Out of Memory!\n"
                f"{'='*60}\n"
                f"The model ran out of GPU memory during generation.\n\n"
                f"Solutions:\n"
                f"1. Use a quantized model variant (A14B-high-quant or A14B-low-quant)\n"
                f"2. Reduce num_frames (try {num_frames // 2})\n"
                f"3. Use 480p instead of 720p resolution\n"
                f"4. Close other GPU applications\n"
                f"5. Use fewer sampling steps\n"
                f"{'='*60}\n"
            )
            print(error_msg)
            clear_cuda_cache()
            raise RuntimeError(error_msg) from e

        except Exception as e:
            error_msg = (
                f"\n{'='*60}\n"
                f"ERROR: Generation failed!\n"
                f"{'='*60}\n"
                f"{str(e)}\n"
                f"{'='*60}\n"
            )
            print(error_msg)
            clear_cuda_cache()
            raise RuntimeError(error_msg) from e

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Determine if the node needs to be re-executed.
        Always re-execute when seed or other parameters change.
        """
        # Return seed to trigger re-execution when seed changes
        return kwargs.get("seed", 0)

    @classmethod
    def VALIDATE_INPUTS(cls, model, image, num_steps, seed, **kwargs):
        """Validate input parameters."""
        if not isinstance(model, dict):
            return "model must be a TurboDiffusion model from TurboWan2ModelLoader"

        if not isinstance(image, torch.Tensor):
            return "image must be a ComfyUI IMAGE tensor"

        if image.dim() != 4 or image.shape[-1] != 3:
            return f"image must be in format [B,H,W,3], got shape {image.shape}"

        if not (1 <= num_steps <= 4):
            return f"num_steps must be between 1 and 4, got {num_steps}"

        num_frames = kwargs.get("num_frames", 77)
        if not (9 <= num_frames <= 121):
            return f"num_frames must be between 9 and 121, got {num_frames}"

        return True

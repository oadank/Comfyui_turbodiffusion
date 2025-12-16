"""TurboWan Image-to-Video Sampler node for ComfyUI.

This node prepares conditioning and latents for TurboDiffusion I2V generation.
Works with standard ComfyUI nodes: UNETLoader, CLIPLoader, VAELoader, CLIPTextEncode, KSamplerAdvanced.

Based on ComfyUI's WanImageToVideo implementation.
"""

from typing import Tuple
import torch
import node_helpers
import comfy.model_management
import comfy.utils


class TurboWanSampler:
    """
    ComfyUI node for TurboWan I2V generation setup.

    Similar to WanImageToVideo, this node:
    - Takes positive/negative conditioning from CLIPTextEncode
    - Takes VAE from VAELoader
    - Takes input image
    - Returns modified conditioning and initial latent for KSamplerAdvanced
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the node."""
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning from CLIP Text Encode"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning from CLIP Text Encode"}),
                "vae": ("VAE", {"tooltip": "VAE from VAE Loader"}),
                "width": ("INT", {
                    "default": 480,
                    "min": 64,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Video width in pixels"
                }),
                "height": ("INT", {
                    "default": 480,
                    "min": 64,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Video height in pixels"
                }),
                "length": ("INT", {
                    "default": 121,
                    "min": 9,
                    "max": 241,
                    "step": 8,
                    "tooltip": "Number of frames to generate"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Batch size for generation"
                }),
            },
            "optional": {
                "start_image": ("IMAGE", {"tooltip": "Optional starting image for I2V"}),
                "clip_vision_output": ("CLIP_VISION_OUTPUT", {"tooltip": "Optional CLIP vision conditioning"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "prepare_conditioning"
    CATEGORY = "video/turbodiffusion"
    DESCRIPTION = "Prepare conditioning and latents for TurboWan I2V generation (use with KSamplerAdvanced)"

    def prepare_conditioning(
        self,
        positive: list,
        negative: list,
        vae,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        start_image=None,
        clip_vision_output=None,
    ) -> Tuple[list, list, dict]:
        """
        Prepare conditioning and initial latent for TurboDiffusion I2V.

        This implementation follows ComfyUI's WanImageToVideo pattern:
        1. Create empty latent tensor for video frames
        2. If start_image provided, encode it and add to conditioning as concat_latent_image
        3. Optionally add CLIP vision conditioning
        4. Return modified conditioning and latent dict for KSamplerAdvanced

        Args:
            positive: Positive conditioning from CLIPTextEncode
            negative: Negative conditioning from CLIPTextEncode
            vae: VAE model
            width: Video width
            height: Video height
            length: Number of frames
            batch_size: Batch size
            start_image: Optional starting image
            clip_vision_output: Optional CLIP vision conditioning

        Returns:
            Tuple of (positive_conditioning, negative_conditioning, latent_dict)
        """
        print(f"\n{'='*60}")
        print(f"TurboWan Sampler - Preparing I2V Generation")
        print(f"{'='*60}")
        print(f"Resolution: {width}x{height}")
        print(f"Frames: {length}")
        print(f"Batch size: {batch_size}")

        # Create empty latent tensor
        # Shape: [batch, channels, temporal, height/8, width/8]
        # For Wan models, latent channels = 16, temporal dimension = ((length - 1) // 4) + 1
        latent = torch.zeros(
            [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=comfy.model_management.intermediate_device()
        )
        print(f"Created latent shape: {latent.shape}")

        # Process start image if provided
        if start_image is not None:
            print(f"Processing start image: {start_image.shape}")

            # Resize start image to target resolution
            # ComfyUI format: [frames, height, width, channels]
            # Need to move channels for upscaling: [frames, channels, height, width]
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center"
            ).movedim(1, -1)

            # Create full-length image sequence filled with gray (0.5)
            # This ensures we have images for all frames
            image = torch.ones(
                (length, height, width, start_image.shape[-1]),
                device=start_image.device,
                dtype=start_image.dtype
            ) * 0.5

            # Place actual start image frames at the beginning
            image[:start_image.shape[0]] = start_image

            # Encode image sequence to latent space using VAE
            concat_latent_image = vae.encode(image[:, :, :, :3])  # Only RGB channels
            print(f"Encoded start image to latent: {concat_latent_image.shape}")

            # Create mask to indicate which frames are from start image vs generated
            # Mask shape: [1, 1, temporal_latent, height_latent, width_latent]
            mask = torch.ones(
                (1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
                device=start_image.device,
                dtype=start_image.dtype
            )
            # Set mask to 0 for start image frames (these are known/fixed)
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            # Add encoded start image and mask to conditioning
            # This tells the model which parts to keep fixed
            positive = node_helpers.conditioning_set_values(
                positive,
                {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative,
                {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )
            print(f"Added start image conditioning with mask")

        # Add CLIP vision conditioning if provided
        if clip_vision_output is not None:
            print(f"Adding CLIP vision conditioning")
            positive = node_helpers.conditioning_set_values(
                positive,
                {"clip_vision_output": clip_vision_output}
            )
            negative = node_helpers.conditioning_set_values(
                negative,
                {"clip_vision_output": clip_vision_output}
            )

        # Prepare output latent dict for KSamplerAdvanced
        out_latent = {"samples": latent}

        print(f"Final latent shape: {latent.shape}")
        print(f"Conditioning prepared successfully")
        print(f"{'='*60}\n")

        return (positive, negative, out_latent)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution when inputs change."""
        # Create hash of important parameters
        return f"{kwargs.get('width', 0)}-{kwargs.get('height', 0)}-{kwargs.get('length', 0)}"

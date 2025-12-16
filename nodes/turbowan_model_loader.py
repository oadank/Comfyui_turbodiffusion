"""Custom model loader for TurboDiffusion quantized .pth files."""

import torch
import folder_paths
import comfy.sd
import comfy.model_management


class TurboWanModelLoader:
    """
    Custom loader for TurboDiffusion quantized .pth models.

    These models use a custom int8 quantization format with separate weight and scale tensors.
    Standard UNETLoader can't load them, so we need this custom loader.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"
    DESCRIPTION = "Load TurboDiffusion quantized models (.pth files with int8_weight format)"

    def load_model(self, model_name):
        """
        Load a TurboDiffusion quantized model.

        The quantized models have keys like:
        - blocks.0.ffn.0.int8_weight (quantized weights)
        - blocks.0.ffn.0.scale (scaling factors)
        - blocks.0.ffn.0.bias (biases)

        We need to dequantize them to standard float format for ComfyUI.
        """
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)

        print(f"\n{'='*60}")
        print(f"Loading TurboDiffusion Quantized Model")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Path: {model_path}")

        # Load the checkpoint
        sd = torch.load(model_path, map_location="cpu", weights_only=False)

        print(f"Loaded checkpoint with {len(sd)} keys")

        # Check if this is a quantized model
        has_quantized_weights = any("int8_weight" in k for k in sd.keys())

        if has_quantized_weights:
            print("Detected quantized model - dequantizing...")
            sd = self._dequantize_state_dict(sd)
            print(f"Dequantized to {len(sd)} keys")
        else:
            print("Model is not quantized, loading directly")

        # Now load using ComfyUI's standard loader
        try:
            model = comfy.sd.load_diffusion_model_state_dict(
                sd,
                model_options={}
            )
            print(f"Successfully loaded model")
            print(f"{'='*60}\n")
            return (model,)
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"{'='*60}\n")
            raise

    def _dequantize_state_dict(self, sd: dict) -> dict:
        """
        Dequantize the state dict from int8_weight + scale format to standard float weights.

        The TurboDiffusion quantization uses block-wise quantization:
        - int8_weight: [M, N] quantized weights
        - scale: [M/block_size, N/block_size] scale factors

        Each block of weights shares a single scale value.
        """
        new_sd = {}
        processed_keys = set()

        for key in sd.keys():
            if key in processed_keys:
                continue

            if key.endswith(".int8_weight"):
                # This is a quantized weight - dequantize it
                base_key = key[:-len(".int8_weight")]
                scale_key = f"{base_key}.scale"
                bias_key = f"{base_key}.bias"
                weight_key = f"{base_key}.weight"

                int8_weight = sd[key]
                scale = sd.get(scale_key)

                if scale is None:
                    # No scale found, just convert to float
                    new_sd[weight_key] = int8_weight.float()
                else:
                    # Block-wise dequantization
                    # Determine block size from shapes
                    weight_shape = int8_weight.shape
                    scale_shape = scale.shape

                    # Calculate block size for each dimension
                    block_sizes = tuple(w // s for w, s in zip(weight_shape, scale_shape))

                    # Upscale the scale tensor to match weight tensor
                    # Using repeat_interleave to replicate each scale value for its block
                    upscaled_scale = scale
                    for dim, block_size in enumerate(block_sizes):
                        upscaled_scale = upscaled_scale.repeat_interleave(block_size, dim=dim)

                    # Now dequantize
                    new_sd[weight_key] = int8_weight.float() * upscaled_scale

                # Copy bias if it exists
                if bias_key in sd:
                    new_sd[bias_key] = sd[bias_key]

                processed_keys.add(key)
                processed_keys.add(scale_key)
                processed_keys.add(bias_key)

            elif key.endswith(".scale"):
                # Scale tensors are handled with their corresponding weights
                processed_keys.add(key)

            elif not key.endswith(".int8_weight") and key not in processed_keys:
                # Regular tensor (not quantized), copy as-is
                new_sd[key] = sd[key]
                processed_keys.add(key)

        return new_sd

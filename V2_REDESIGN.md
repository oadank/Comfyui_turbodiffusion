# V2.0 Redesign - Full ComfyUI Native Integration

## Summary

Complete redesign of TurboDiffusion nodes to follow ComfyUI's standard workflow patterns, matching the official Wan2.2 workflow structure.

## Key Changes

### 1. Removed Custom Loaders ❌

**Deleted:**
- `nodes/model_loader.py` (TurboWan2ModelLoader)
- `nodes/vae_loader.py` (TurboWanVAELoader)
- `nodes/t5_loader.py` (TurboWanT5Loader)
- `nodes/i2v_generator.py` (TurboDiffusionI2V)

**Reason:** These duplicated ComfyUI's built-in functionality. Users should use standard loaders.

### 2. New Simplified Node Structure ✓

**Added:**
- `nodes/turbowan_sampler.py` (TurboWanSampler) - Core I2V preparation node
- `nodes/video_saver.py` (TurboDiffusionSaveVideo) - Kept for video export convenience

**Total Nodes: 2** (down from 5)

### 3. Use Standard ComfyUI Nodes

| Purpose | Node to Use | Type |
|---------|-------------|------|
| Load High Noise Model | UNETLoader | Built-in |
| Load Low Noise Model | UNETLoader | Built-in |
| Load Text Encoder | CLIPLoader | Built-in |
| Load VAE | VAELoader | Built-in |
| Positive Prompt | CLIPTextEncode | Built-in |
| Negative Prompt | CLIPTextEncode | Built-in |
| Prepare I2V | TurboWanSampler | **Custom** |
| Sampling Config | ModelSamplingSD3 | Built-in |
| Dual-Expert Sampling | KSamplerAdvanced (×2) | Built-in |
| Decode Latents | VAEDecode | Built-in |
| Preview | PreviewImage | Built-in |
| Save Video | TurboDiffusionSaveVideo | **Custom** |

### 4. TurboWanSampler Node

Replaces the old monolithic `TurboDiffusionI2V` node.

**Design Pattern:** Follows `WanImageToVideo` from official Wan workflows

**Inputs:**
```python
{
    "required": {
        "positive": CONDITIONING,      # From CLIPTextEncode
        "negative": CONDITIONING,      # From CLIPTextEncode
        "vae": VAE,                   # From VAELoader
        "width": INT,                 # Video width
        "height": INT,                # Video height
        "length": INT,                # Number of frames
        "batch_size": INT,            # Batch size
    },
    "optional": {
        "start_image": IMAGE,         # Starting frame
        "clip_vision_output": CLIP_VISION_OUTPUT,
    }
}
```

**Outputs:**
```python
(
    positive_conditioning,  # Modified for video
    negative_conditioning,  # Modified for video
    latent_dict,           # Initial latent for KSamplerAdvanced
)
```

**Purpose:**
- Prepares video-specific conditioning
- Creates initial latent tensor (batch × 4 × frames × height/8 × width/8)
- Encodes start image if provided
- Adds temporal/frame metadata to conditioning

### 5. Updated Workflow Structure

**Old Approach (v0.1):**
```
Custom Loaders → Custom I2V Node → Save Video
```

**New Approach (v0.2):**
```
Step 1: Standard Loaders
├─ UNETLoader (high noise)
├─ UNETLoader (low noise)
├─ CLIPLoader (umT5)
└─ VAELoader

Step 2: Standard Prompts
├─ CLIPTextEncode (positive)
└─ CLIPTextEncode (negative)

Step 3: I2V Preparation
└─ TurboWanSampler

Step 4: Standard Sampling
├─ ModelSamplingSD3 (high)
├─ KSamplerAdvanced (stage 1: steps 0-2)
├─ ModelSamplingSD3 (low)
└─ KSamplerAdvanced (stage 2: steps 2-4)

Step 5: Standard Decode
├─ VAEDecode
├─ PreviewImage
└─ Save Video (custom)
```

### 6. Workflow File Changes

**Removed:**
- `turbodiffusion_workflow_single_model.json`
- `turbodiffusion_workflow_dual_expert.json`
- `turbodiffusion_workflow_native.json`

**Added:**
- `turbowan_workflow.json` - Single comprehensive workflow with clear labeling

### 7. Model File Locations

**No Change** - Still uses standard ComfyUI folders:
```
ComfyUI/models/
├─ diffusion_models/
│  ├─ wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
│  └─ wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
├─ text_encoders/
│  └─ nsfw_wan_umt5-xxl_fp8_scaled.safetensors
└─ vae/
   └─ wan_2.1_vae.safetensors
```

## Benefits of V2 Design

### 1. True ComfyUI Native Integration
- Uses standard nodes everywhere possible
- Follows established ComfyUI patterns
- No custom reinvention of existing functionality

### 2. Consistent with Official Workflows
- Matches Wan2.2 workflow structure exactly
- Same node types and connections
- Familiar to users of official Wan workflows

### 3. Simplified Maintenance
- Only 2 custom nodes instead of 5
- Less code to maintain
- Easier to debug and update

### 4. Better User Experience
- Familiar workflow for ComfyUI users
- Can swap components easily (try different models, VAEs, prompts)
- Standard nodes have built-in features (model auto-download, etc.)

### 5. Flexibility
- Users can customize any part of the workflow
- Can use different text encoders, VAEs, samplers
- Easy to integrate with other custom nodes

## Migration Guide

### From V0.1 to V0.2

**Old Workflow:**
```json
{
  "TurboWan2ModelLoader": {
    "model_filename": "TurboWan2.2-I2V-A14B-high-720P-quant.pth",
    "vae": null,  // auto-loads
    "text_encoder": null  // auto-loads
  }
}
```

**New Workflow:**
```json
{
  "UNETLoader": {
    "unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    "weight_dtype": "fp8_e4m3fn"
  },
  "CLIPLoader": {
    "clip_name": "nsfw_wan_umt5-xxl_fp8_scaled.safetensors",
    "type": "wan"
  },
  "VAELoader": {
    "vae_name": "wan_2.1_vae.safetensors"
  }
}
```

### Model Filename Changes

| Old (v0.1) | New (v0.2) |
|------------|------------|
| TurboWan2.2-I2V-A14B-high-720P-quant.pth | wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors |
| TurboWan2.2-I2V-A14B-low-720P-quant.pth | wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors |
| Wan2.1_VAE.pth | wan_2.1_vae.safetensors |
| models_t5_umt5-xxl-enc-bf16.pth | nsfw_wan_umt5-xxl_fp8_scaled.safetensors |

**Note:** Use the official Wan2.2 ComfyUI repackaged models from:
https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged

## Implementation Status

✅ TurboWanSampler node created
✅ Node registration updated
✅ Workflow created with proper labeling
✅ README updated
✅ Old custom loaders removed
✅ Version bumped to 0.2.0

⚠️ TODO: Implement actual TurboDiffusion inference in TurboWanSampler.prepare_conditioning()
⚠️ TODO: Add proper image encoding to latent
⚠️ TODO: Add temporal conditioning modifications

## Testing

```bash
# Verify nodes load
cd ComfyUI/custom_nodes
python -c "import comfyui_turbodiffusion; print(comfyui_turbodiffusion.NODE_CLASS_MAPPINGS.keys())"

# Expected output:
# dict_keys(['TurboWanSampler', 'TurboDiffusionSaveVideo'])
```

## Documentation Updates

- ✅ README.md - Complete rewrite
- ✅ __init__.py - Updated docstring and node list
- ✅ V2_REDESIGN.md - This document
- ⚠️ INSTALLATION.md - Needs update for new workflow
- ⚠️ QUICKSTART.md - Needs update for new node structure

## Next Steps

1. Implement actual TurboDiffusion inference logic in TurboWanSampler
2. Test with real models and verify video generation
3. Update remaining documentation
4. Create tutorial video/screenshots
5. Submit to ComfyUI Manager registry

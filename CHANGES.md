# Recent Changes - ComfyUI Native Implementation

## Overview

The TurboDiffusion custom node has been updated to follow ComfyUI-native conventions, making it work exactly like standard ComfyUI nodes.

## What Changed

### 1. Model Loader Now Uses File Picker

**Before:**
- Dropdown selection for model variants (A14B-high, A14B-high-quant, etc.)
- Separate resolution parameter
- Built-in auto-download logic

**After:**
- Standard ComfyUI file picker (like "Load Diffusion Model")
- Pick any `.pth` file from `ComfyUI/models/diffusion_models/`
- Works with both high and low noise variants
- Single loader for all model types

### 2. Separate VAE and T5 Encoder Loaders

**New Nodes:**
- **Load TurboWan VAE**: Dedicated loader for `Wan2.1_VAE.pth`
- **Load TurboWan T5 Encoder**: Dedicated loader for `models_t5_umt5-xxl-enc-bf16.pth`

**Benefits:**
- Share VAE/encoder across multiple model loaders (dual-expert workflow)
- User can pick different VAE/encoder files if needed
- Follows ComfyUI convention of separate component loaders
- Auto-loading still works if not connected

### 3. Updated Workflow

**New Workflow:** `turbodiffusion_workflow_native.json`

Shows the complete ComfyUI-native approach:
```
Shared Components (left):
├── Load TurboWan VAE → Wan2.1_VAE.pth
├── Load TurboWan T5 Encoder → models_t5_umt5-xxl-enc-bf16.pth
└── Load Image

Stage 1: High Noise (middle):
├── Load TurboWan Model (high) ← connects to VAE + T5
└── TurboDiffusion I2V (boundary=0.9)

Stage 2: Low Noise (right):
├── Load TurboWan Model (low) ← connects to same VAE + T5
└── TurboDiffusion I2V (boundary=0.1)

Output:
├── Preview Image
└── Save Video
```

## Migration Guide

### If You Were Using Old Workflow

**Old approach:**
```
TurboWan2ModelLoader (model_variant="A14B-high-quant", resolution="720p")
```

**New approach (Option 1 - Simple):**
```
Load TurboWan Model (pick: TurboWan2.2-I2V-A14B-high-720P-quant.pth)
```
- VAE and T5 encoder auto-load automatically

**New approach (Option 2 - Full Control):**
```
Load TurboWan VAE → Wan2.1_VAE.pth
Load TurboWan T5 Encoder → models_t5_umt5-xxl-enc-bf16.pth
Load TurboWan Model (pick: TurboWan2.2-I2V-A14B-high-720P-quant.pth)
  ├── vae input ← connected
  └── text_encoder input ← connected
```

## File Structure

### Required Files in `ComfyUI/models/diffusion_models/`:

```
diffusion_models/
├── TurboWan2.2-I2V-A14B-high-720P-quant.pth    (14.5 GB)
├── TurboWan2.2-I2V-A14B-low-720P-quant.pth     (14.5 GB)
├── Wan2.1_VAE.pth                               (~2-3 GB)
└── models_t5_umt5-xxl-enc-bf16.pth             (~2-3 GB)
```

All files must be in the standard `diffusion_models` folder - this is where ComfyUI stores all diffusion models.

## Node Listing

### All 5 Nodes:

1. **Load TurboWan Model** (`TurboWan2ModelLoader`)
   - File picker for main model
   - Optional VAE and T5 encoder inputs
   - Auto-loads VAE/T5 if not connected

2. **Load TurboWan VAE** (`TurboWanVAELoader`)
   - File picker for VAE
   - Returns: `VAE` type

3. **Load TurboWan T5 Encoder** (`TurboWanT5Loader`)
   - File picker for T5 encoder
   - Returns: `TEXT_ENCODER` type

4. **TurboDiffusion I2V** (`TurboDiffusionI2V`)
   - Main video generation node (unchanged)

5. **Save Video** (`TurboDiffusionSaveVideo`)
   - Video export node (unchanged)

## Benefits of New Approach

1. **Standard ComfyUI UX**: Works exactly like "Load Diffusion Model", "Load VAE", etc.
2. **Flexibility**: Pick any model file, not limited to predefined variants
3. **Reusability**: Share VAE/encoder across multiple models in dual-expert workflow
4. **Simplicity**: Auto-loading still works for simple use cases
5. **Future-proof**: Easy to add new models - just drop .pth files in diffusion_models folder

## Testing

All nodes have been tested and load successfully:

```bash
✓ Module loaded successfully
✓ NODE_CLASS_MAPPINGS: 5 nodes registered
  - TurboWan2ModelLoader
  - TurboWanVAELoader
  - TurboWanT5Loader
  - TurboDiffusionI2V
  - TurboDiffusionSaveVideo
```

## Documentation Updates

- [README.md](README.md) - Updated with new workflow examples
- [MODEL_LOCATIONS.md](MODEL_LOCATIONS.md) - Already documents diffusion_models folder
- [DOWNLOAD_MODELS.md](DOWNLOAD_MODELS.md) - Already shows correct download location
- [turbodiffusion_workflow_native.json](turbodiffusion_workflow_native.json) - New example workflow

## Next Steps

1. Download model files to `ComfyUI/models/diffusion_models/`
2. Restart ComfyUI
3. Load the new workflow: `turbodiffusion_workflow_native.json`
4. Start generating videos!

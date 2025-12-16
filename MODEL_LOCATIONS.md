# TurboDiffusion Model File Locations

## ✅ Recommended: ComfyUI Standard Location

Place your TurboDiffusion models in ComfyUI's standard diffusion models folder:

```
C:\Users\Ganaraj\Downloads\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI\models\diffusion_models\
```

### Why This Location?

1. **Standard ComfyUI convention** - All diffusion models go in `models/diffusion_models/`
2. **Shared across nodes** - Other custom nodes can also access models here
3. **Organized** - Keep all your diffusion models (SD, Flux, TurboDiffusion, etc.) in one place
4. **ComfyUI-Manager compatible** - Follows the standard folder structure

### What to Put There:

```
diffusion_models/
├── TurboWan2.2-I2V-A14B-high-720P-quant.pth      (14.5 GB)
├── TurboWan2.2-I2V-A14B-low-720P-quant.pth       (14.5 GB)
├── Wan2.1_VAE.pth                                 (~2-3 GB)
├── models_t5_umt5-xxl-enc-bf16.pth               (~2-3 GB)
└── [your other diffusion models...]
```

## 🔄 Alternative: Node's Local Folder

The node can also use its own local folder:

```
C:\Users\Ganaraj\Documents\Projects\comfyui-turbodiffusion\checkpoints\
```

This is useful if:
- You want to keep TurboDiffusion models separate
- You're testing/developing the node
- You have models in multiple locations

## 🔍 Search Order

The node searches for models in this order:

1. **ComfyUI/models/diffusion_models/** ← Checks here first
2. **custom_nodes/comfyui-turbodiffusion/checkpoints/** ← Falls back to here

The first location where a model is found will be used.

## 📦 Quick Setup

### Option 1: Copy to ComfyUI Models Folder

If you've already downloaded models:

```bash
# Copy models to ComfyUI standard location
cp *.pth "C:\Users\Ganaraj\Downloads\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI\models\diffusion_models\"
```

### Option 2: Download Directly to ComfyUI

```bash
# Navigate to ComfyUI models folder
cd "C:\Users\Ganaraj\Downloads\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI\models\diffusion_models"

# Download using HuggingFace CLI
huggingface-cli download TurboDiffusion/TurboWan2.2-I2V-A14B-720P TurboWan2.2-I2V-A14B-high-720P-quant.pth --local-dir .
huggingface-cli download TurboDiffusion/TurboWan2.2-I2V-A14B-720P TurboWan2.2-I2V-A14B-low-720P-quant.pth --local-dir .
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B Wan2.1_VAE.pth --local-dir .
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B models_t5_umt5-xxl-enc-bf16.pth --local-dir .
```

## ✅ Verify Installation

Run the verification script:

```bash
cd "C:\Users\Ganaraj\Downloads\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI"
python custom_nodes/comfyui-turbodiffusion/test_installation.py
```

Or manually check:

```bash
ls "C:\Users\Ganaraj\Downloads\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI\models\diffusion_models\" | grep -i turbo
```

You should see:
- TurboWan2.2-I2V-A14B-high-720P-quant.pth
- TurboWan2.2-I2V-A14B-low-720P-quant.pth
- Wan2.1_VAE.pth
- models_t5_umt5-xxl-enc-bf16.pth

## 📖 Additional Resources

- **Full download guide**: [DOWNLOAD_MODELS.md](DOWNLOAD_MODELS.md)
- **Installation guide**: [INSTALLATION.md](INSTALLATION.md)
- **Quick start**: [QUICKSTART.md](QUICKSTART.md)

## 💡 Tips

- **Use quantized models** (`-quant.pth`) for 24GB GPUs - they're half the size!
- **Download VAE and T5 first** - they're smaller and faster to download
- **Keep models in diffusion_models/** - follows ComfyUI best practices
- **Don't duplicate** - pick one location and stick with it

## 🆘 If Node Can't Find Models

The node will print where it's searching in the ComfyUI console:

```
Looking for models in:
  1. C:\Users\Ganaraj\Downloads\...\ComfyUI\models\diffusion_models\
  2. C:\Users\Ganaraj\Documents\Projects\comfyui-turbodiffusion\checkpoints\
```

Check the console output to see which paths it's actually searching!

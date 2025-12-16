# Model Download Guide

This guide shows you how to manually download the TurboWan2.2 model files.

## 📥 Download Location

**Option 1: ComfyUI Standard Diffusion Models Folder (RECOMMENDED)**
```
C:\Users\Ganaraj\Downloads\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI\models\diffusion_models\
```
This is the standard location where ComfyUI stores all diffusion models, following ComfyUI conventions.

**Option 2: Node's Local Checkpoints Folder**
```
C:\Users\Ganaraj\Documents\Projects\comfyui-turbodiffusion\checkpoints\
```

The node automatically searches both locations, preferring the ComfyUI standard `diffusion_models` folder. This allows TurboDiffusion models to be stored alongside other diffusion models like Stable Diffusion, Flux, etc.

## 🔗 Download Links

### Main Models (TurboWan2.2-I2V-A14B-720P)

**Repository**: https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/tree/main

#### High Noise Model (choose one):

**Option 1: Quantized (14.5 GB) - RECOMMENDED for 24GB GPUs**
- File: `TurboWan2.2-I2V-A14B-high-720P-quant.pth`
- Direct link: https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-high-720P-quant.pth

**Option 2: Full Precision (28.6 GB) - For 40GB+ GPUs**
- File: `TurboWan2.2-I2V-A14B-high-720P.pth`
- Direct link: https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-high-720P.pth

#### Low Noise Model (choose one):

**Option 1: Quantized (14.5 GB) - RECOMMENDED for 24GB GPUs**
- File: `TurboWan2.2-I2V-A14B-low-720P-quant.pth`
- Direct link: https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-low-720P-quant.pth

**Option 2: Full Precision (28.6 GB) - For 40GB+ GPUs**
- File: `TurboWan2.2-I2V-A14B-low-720P.pth`
- Direct link: https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-low-720P.pth

### Required Components (Wan2.1)

**Repository**: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main

#### VAE (Required)
- File: `Wan2.1_VAE.pth`
- Direct link: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth

#### T5 Text Encoder (Required)
- File: `models_t5_umt5-xxl-enc-bf16.pth`
- Direct link: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth

## 📋 Download Checklist

For **24GB GPU** (RTX 4090/5090):
- [ ] `TurboWan2.2-I2V-A14B-high-720P-quant.pth` (14.5 GB)
- [ ] `TurboWan2.2-I2V-A14B-low-720P-quant.pth` (14.5 GB)
- [ ] `Wan2.1_VAE.pth` (~2-3 GB)
- [ ] `models_t5_umt5-xxl-enc-bf16.pth` (~2-3 GB)
- **Total**: ~33 GB

For **40GB+ GPU** (H100/A100):
- [ ] `TurboWan2.2-I2V-A14B-high-720P.pth` (28.6 GB)
- [ ] `TurboWan2.2-I2V-A14B-low-720P.pth` (28.6 GB)
- [ ] `Wan2.1_VAE.pth` (~2-3 GB)
- [ ] `models_t5_umt5-xxl-enc-bf16.pth` (~2-3 GB)
- **Total**: ~62 GB

## 💻 Download Methods

### Method 1: Browser Download

1. Click the direct links above
2. Browser will start downloading
3. Move downloaded files to checkpoints folder

### Method 2: wget (if available)

```bash
cd C:\Users\Ganaraj\Documents\Projects\comfyui-turbodiffusion\checkpoints\

# High noise (quantized)
wget https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-high-720P-quant.pth

# Low noise (quantized)
wget https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-low-720P-quant.pth

# VAE
wget https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth

# T5 Encoder
wget https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
```

### Method 3: HuggingFace CLI

```bash
# Install huggingface-hub CLI (if not installed)
pip install huggingface-hub[cli]

# Login (if needed)
huggingface-cli login

# Download files
cd C:\Users\Ganaraj\Documents\Projects\comfyui-turbodiffusion\checkpoints\

huggingface-cli download TurboDiffusion/TurboWan2.2-I2V-A14B-720P TurboWan2.2-I2V-A14B-high-720P-quant.pth --local-dir .
huggingface-cli download TurboDiffusion/TurboWan2.2-I2V-A14B-720P TurboWan2.2-I2V-A14B-low-720P-quant.pth --local-dir .
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B Wan2.1_VAE.pth --local-dir .
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B models_t5_umt5-xxl-enc-bf16.pth --local-dir .
```

### Method 4: Git LFS (Full Repository)

```bash
# Install git-lfs first
git lfs install

# Clone repositories
cd C:\Users\Ganaraj\Documents\Projects\comfyui-turbodiffusion\checkpoints\

# Clone TurboWan models
git clone https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P
# Then copy the .pth files from the cloned folder to checkpoints/

# Clone Wan2.1 models
git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
# Then copy the required .pth files to checkpoints/
```

## ✅ Verify Downloads

After downloading, verify the files are in the correct location:

### Windows Command Prompt:
```cmd
dir C:\Users\Ganaraj\Documents\Projects\comfyui-turbodiffusion\checkpoints\
```

### Git Bash / WSL:
```bash
ls -lh /c/Users/Ganaraj/Documents/Projects/comfyui-turbodiffusion/checkpoints/
```

You should see:
```
TurboWan2.2-I2V-A14B-high-720P-quant.pth    (14.5 GB or 28.6 GB)
TurboWan2.2-I2V-A14B-low-720P-quant.pth     (14.5 GB or 28.6 GB)
Wan2.1_VAE.pth                               (~2-3 GB)
models_t5_umt5-xxl-enc-bf16.pth             (~2-3 GB)
```

## 🚀 After Download

Once all files are downloaded:

1. **Disable auto-download** in the Model Loader node (optional):
   - Set `auto_download` to `False` in the node
   - Or leave it as `True` - it will skip downloading if files exist

2. **Use the models** in ComfyUI:
   - Load the TurboWan2.2 Model Loader node
   - Select your variant (A14B-high-quant or A14B-low-quant)
   - The node will use your downloaded files automatically

3. **Test the setup**:
   - Load example workflow
   - Connect nodes
   - Run generation

## 💡 Tips

- **Start with quantized models** (14.5 GB) to save space and VRAM
- **Download VAE and T5 encoder first** - they're smaller and required for both variants
- **Use a download manager** for large files to support resume on failure
- **Check disk space** before starting (need ~35GB free for quantized, ~65GB for full)
- **Keep files in checkpoints/** - the node automatically looks there first

## 🆘 Troubleshooting

### Download fails/corrupts
- Use a download manager that supports resume (e.g., Free Download Manager)
- Check your internet connection stability
- Verify disk space before downloading

### Files are in wrong location
```bash
# Move files to correct location
mv /path/to/downloaded/*.pth C:\Users\Ganaraj\Documents\Projects\comfyui-turbodiffusion\checkpoints\
```

### Node can't find files
- Verify files are directly in `checkpoints/` folder (not in a subfolder)
- Check filenames match exactly (case-sensitive)
- Ensure files are `.pth` format (not `.zip` or incomplete downloads)

### Still want auto-download
- Delete manually downloaded files
- Enable `auto_download` in Model Loader node
- Node will download fresh copies from HuggingFace

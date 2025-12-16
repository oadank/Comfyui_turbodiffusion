# Installation Guide for ComfyUI TurboDiffusion

This guide explains how to install and use the TurboDiffusion I2V custom node in your ComfyUI installation.

## Prerequisites

Before installing, ensure you have:

1. **ComfyUI** installed and working
2. **Python 3.9+** (Python 3.11 recommended)
3. **CUDA-capable GPU** with 24GB+ VRAM (RTX 4090/5090 or better)
4. **PyTorch 2.7.0+** (2.8.0 recommended) with CUDA support
5. **uv** package manager installed (optional but recommended)

## Installation Methods

### Method 1: Using uv (Recommended)

1. **Navigate to ComfyUI custom_nodes directory:**
   ```bash
   cd /path/to/ComfyUI/custom_nodes/
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/comfyui-turbodiffusion.git
   cd comfyui-turbodiffusion
   ```

3. **Install dependencies with uv:**
   ```bash
   # Install uv if you don't have it
   pip install uv

   # Install all dependencies
   uv sync

   # Optional: Install video export support
   uv sync --extra video

   # Optional: Install SageSLA attention support
   uv sync --extra sagesla
   ```

4. **Restart ComfyUI**

### Method 2: Using pip

1. **Navigate to ComfyUI custom_nodes directory:**
   ```bash
   cd /path/to/ComfyUI/custom_nodes/
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/comfyui-turbodiffusion.git
   cd comfyui-turbodiffusion
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .

   # Optional: Install video export support
   pip install -e ".[video]"

   # Optional: Install SageSLA support
   pip install -e ".[sagesla]"
   ```

4. **Restart ComfyUI**

### Method 3: Manual Installation (Advanced)

If you want to manually install dependencies:

```bash
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/your-org/comfyui-turbodiffusion.git
cd comfyui-turbodiffusion

# Install required packages
pip install torch>=2.7.0
pip install turbodiffusion --no-build-isolation
pip install huggingface-hub>=0.20.0
pip install Pillow>=10.0.0
pip install numpy>=1.24.0
pip install tqdm>=4.65.0

# Optional: Video export
pip install opencv-python imageio imageio-ffmpeg

# Optional: SageSLA attention
pip install git+https://github.com/thu-ml/SpargeAttn.git
```

## Verifying Installation

After installation and restarting ComfyUI:

1. **Open ComfyUI in your browser**
2. **Right-click on the canvas** to open the node menu
3. **Look for the category:** `video/turbodiffusion`
4. **You should see 3 nodes:**
   - TurboWan2.2 Model Loader
   - TurboDiffusion I2V Generator
   - Save TurboDiffusion Video

If you see these nodes, the installation was successful!

## First Run Setup

### Automatic Model Download (Recommended)

On first use, the node will automatically download required models (~15GB total):

1. **Add "TurboWan2.2 Model Loader" node**
2. **Set parameters:**
   - model_variant: `A14B-high-quant` (recommended for 24GB GPU)
   - resolution: `720p`
   - auto_download: `True` ✓
3. **Queue the workflow**

The node will download:
- TurboWan2.2-I2V model checkpoint (~14.5GB for quantized)
- Wan2.1 VAE checkpoint (~2-3GB)
- umT5-XXL text encoder (~2-3GB)

Files are saved to: `comfyui-turbodiffusion/checkpoints/`

### Manual Model Download (Alternative)

If you prefer to download models manually:

1. **Download from HuggingFace:**
   - Model: [TurboDiffusion/TurboWan2.2-I2V-A14B-720P](https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P)
   - VAE & Encoder: [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)

2. **Place files in:**
   ```
   comfyui-turbodiffusion/checkpoints/
   ├── TurboWan2.2-I2V-A14B-high-720P-quant.pth
   ├── Wan2.1_VAE.pth
   └── models_t5_umt5-xxl-enc-bf16.pth
   ```

3. **In the Model Loader node:**
   - Set auto_download: `False`
   - Or set checkpoint_path to your custom location

## Example Workflow

1. **Load the example workflow:**
   - Open `example_workflow.json` in ComfyUI
   - Or manually create the following setup:

2. **Node setup:**
   ```
   Load Image → TurboDiffusion I2V → Preview Image
                      ↑                      ↓
             TurboWan2 Model Loader    Save Video (MP4)
   ```

3. **Configure nodes:**
   - **Load Image**: Select your input image
   - **Model Loader**:
     - Variant: A14B-high-quant
     - Resolution: 720p
   - **I2V Generator**:
     - num_steps: 4 (higher quality)
     - num_frames: 77 (3.2s at 24fps)
     - seed: any number for reproducibility
     - prompt: optional text description
   - **Save Video**:
     - fps: 24
     - format: mp4

4. **Queue and wait** (~30-60 seconds on RTX 5090)

## Troubleshooting

### "Module 'turbodiffusion' not found"

```bash
cd /path/to/ComfyUI/custom_nodes/comfyui-turbodiffusion
pip install turbodiffusion --no-build-isolation
```

### "CUDA Out of Memory"

Solutions:
1. Use quantized model: `A14B-high-quant` or `A14B-low-quant`
2. Reduce num_frames: try 49 or 33 instead of 77
3. Use 480p instead of 720p
4. Close other GPU applications
5. Reduce num_steps to 2 or 1

### "Models not downloading"

1. Check internet connection
2. Verify HuggingFace is accessible
3. Try manual download from HuggingFace
4. Check disk space (~20GB free required)

### "Video export fails"

Install video export dependencies:
```bash
pip install opencv-python imageio imageio-ffmpeg
# OR
uv sync --extra video
```

### "Nodes don't appear in ComfyUI"

1. Verify installation location:
   ```bash
   ls /path/to/ComfyUI/custom_nodes/comfyui-turbodiffusion
   ```
2. Check ComfyUI console for errors
3. Ensure __init__.py exists and is not corrupted
4. Restart ComfyUI completely
5. Try running ComfyUI with `--verbose` flag

## GPU Memory Requirements

| Configuration | VRAM Required | GPU Examples |
|--------------|---------------|--------------|
| 480p, Quantized, 49 frames | ~18GB | RTX 4090 |
| 720p, Quantized, 77 frames | ~22GB | RTX 4090, RTX 5090 |
| 720p, Full precision, 77 frames | ~38GB | H100, A100 |
| 720p, Quantized, 121 frames | ~30GB | RTX 5090, A6000 |

## Performance Tips

1. **Use quantized models** for 24GB GPUs
2. **Lower num_steps** for faster generation (quality trade-off)
3. **Use SLA attention** for 2x speedup:
   - Install: `pip install git+https://github.com/thu-ml/SpargeAttn.git`
   - Set attention_type: `sagesla`
4. **Batch generation**: Generate multiple videos in parallel if you have VRAM
5. **Monitor GPU usage**: Use `nvidia-smi` to track VRAM

## Updating

To update to the latest version:

```bash
cd /path/to/ComfyUI/custom_nodes/comfyui-turbodiffusion
git pull
uv sync  # or pip install -e . --upgrade
```

Restart ComfyUI after updating.

## Uninstallation

To remove the custom node:

```bash
cd /path/to/ComfyUI/custom_nodes/
rm -rf comfyui-turbodiffusion
```

Restart ComfyUI.

Note: Downloaded model checkpoints (~15GB) will remain in the checkpoints/ directory. Delete manually if needed.

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/comfyui-turbodiffusion/issues)
- **TurboDiffusion**: [GitHub](https://github.com/thu-ml/TurboDiffusion)
- **ComfyUI**: [GitHub](https://github.com/comfyanonymous/ComfyUI)

## Next Steps

After installation:
1. Try the example workflow
2. Experiment with different parameters
3. Read the main [README.md](README.md) for parameter details
4. Check out advanced features (SageSLA, ODE sampling)

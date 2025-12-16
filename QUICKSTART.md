# Quick Start Guide

Get started with TurboDiffusion I2V in 5 minutes!

## Installation (2 minutes)

```bash
# Navigate to ComfyUI custom_nodes
cd /path/to/ComfyUI/custom_nodes/

# Clone this repository
git clone https://github.com/your-org/comfyui-turbodiffusion.git
cd comfyui-turbodiffusion

# Install dependencies (choose one)
uv sync              # If you have uv (recommended)
# OR
pip install -e .     # Standard pip

# Restart ComfyUI
```

## First Video Generation (3 minutes)

### Option 1: Load Example Workflow

1. Open ComfyUI
2. Load `example_workflow.json` from this repository
3. Update the "Load Image" node with your image path
4. Click "Queue Prompt"
5. Wait ~30-60 seconds
6. Video saves to ComfyUI output folder!

### Option 2: Manual Setup

1. **Add nodes** (right-click → video/turbodiffusion):
   - Load Image (built-in)
   - TurboWan2.2 Model Loader
   - TurboDiffusion I2V Generator
   - Save TurboDiffusion Video

2. **Connect nodes:**
   ```
   Load Image → I2V Generator → Save Video
                     ↑
               Model Loader
   ```

3. **Configure:**
   - **Load Image**: Choose your image
   - **Model Loader**:
     - model_variant: `A14B-high-quant` (14.5GB)
     - resolution: `720p`
     - auto_download: ✓ (first run downloads ~15GB)
   - **I2V Generator**:
     - num_steps: `4` (best quality)
     - num_frames: `77` (3.2s video)
     - seed: `12345` (or any number)
     - prompt: `"smooth motion, cinematic"` (optional)
   - **Save Video**:
     - format: `mp4`
     - fps: `24`

4. **Queue and wait**
   - First run: ~5-10 mins (model download)
   - Subsequent runs: ~30-60 seconds
   - Check ComfyUI console for progress

5. **Find your video:**
   - Location: `ComfyUI/output/turbodiffusion_videos/`
   - Or the path shown in console

## Tips for First-Time Users

### GPU Requirements
- **Minimum**: RTX 4090 (24GB) with quantized model
- **Recommended**: RTX 5090 or better

### If You Get "Out of Memory" Error
1. Switch to `A14B-high-quant` (not full precision)
2. Reduce `num_frames` to `49` or `33`
3. Try `480p` instead of `720p`

### Model Download Takes Forever?
- First download is ~15GB (be patient!)
- Files saved to: `comfyui-turbodiffusion/checkpoints/`
- Reused for all future generations
- Download once, use forever

### Want Faster Generation?
- Set `num_steps` to `1` or `2` (quality trade-off)
- Use `sla` attention type (requires spargeattn)
- Reduce `num_frames`

## Example Results

With default settings (4 steps, 77 frames, 720p):
- **Generation time**: 30-60 seconds (RTX 5090)
- **Video length**: ~3.2 seconds at 24 fps
- **File size**: ~5-10 MB (MP4)
- **Quality**: High-quality, smooth motion

## Next Steps

Once you have your first video:

1. **Experiment with prompts**:
   - "slow zoom in, cinematic"
   - "camera pan left to right"
   - "fast motion, dynamic"

2. **Try different settings**:
   - More frames (121 max) for longer videos
   - Different seeds for variations
   - sigma_max for more/less variation

3. **Chain with other nodes**:
   - Use ControlNet for guided generation
   - Upscale frames before generation
   - Post-process with video filters

4. **Read full documentation**:
   - [README.md](README.md) - Complete parameter reference
   - [INSTALLATION.md](INSTALLATION.md) - Detailed install guide
   - [GitHub Issues](https://github.com/your-org/comfyui-turbodiffusion/issues) - Get help

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| Nodes don't appear | Restart ComfyUI, check console for errors |
| Import error | Run `pip install turbodiffusion --no-build-isolation` |
| CUDA OOM | Use quantized model, reduce frames |
| Slow download | Normal for first time (~15GB), be patient |
| Video won't save | Install: `pip install opencv-python imageio` |

## Support

Questions? Issues?
- [GitHub Issues](https://github.com/your-org/comfyui-turbodiffusion/issues)
- [TurboDiffusion Docs](https://github.com/thu-ml/TurboDiffusion)
- [ComfyUI Discord](https://discord.gg/comfyui)

Happy generating! 🎬

# ComfyUI TurboDiffusion I2V Node

ComfyUI custom node for [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) Image-to-Video (I2V) generation. This node provides a seamless integration of the TurboWan2.2-I2V-A14B model for high-quality, fast video generation from images.

## Features

- **Fast Video Generation**: 100-205× acceleration using TurboDiffusion framework
- **High Quality**: Supports 480p and 720p video generation
- **Flexible Models**: Both quantized (14.5GB) and full precision (28.6GB) models
- **Easy Integration**: Drop-in custom node for existing ComfyUI installations
- **Auto-Download**: Automatic model downloading from HuggingFace
- **Advanced Controls**: Full control over inference parameters (steps, frames, attention types)

## Requirements

### GPU Requirements
- **Minimum**: NVIDIA RTX 4090 (24GB VRAM) with quantized models
- **Recommended**: NVIDIA RTX 5090, H100, or A100
- Higher resolutions and more frames require more VRAM

### Software Requirements
- Python >= 3.9
- PyTorch >= 2.7.0 (2.8.0 recommended)
- ComfyUI (latest version)
- CUDA-capable GPU

## Installation

### Using uv (Recommended)

1. Navigate to your ComfyUI custom_nodes directory:
```bash
cd path/to/ComfyUI/custom_nodes/
```

2. Clone this repository:
```bash
git clone https://github.com/your-org/comfyui-turbodiffusion.git
cd comfyui-turbodiffusion
```

3. Install dependencies with uv:
```bash
uv sync
```

4. (Optional) Install video export support:
```bash
uv sync --extra video
```

5. (Optional) Install SageSLA attention support:
```bash
uv sync --extra sagesla
```

### Using pip

```bash
cd path/to/ComfyUI/custom_nodes/
git clone https://github.com/your-org/comfyui-turbodiffusion.git
cd comfyui-turbodiffusion
pip install -e .
```

### Restart ComfyUI

After installation, restart ComfyUI to load the new nodes.

## Usage

### Available Nodes

The node pack provides 5 ComfyUI-native components:

1. **Load TurboWan Model**: Standard file picker for TurboWan models (high or low noise)
2. **Load TurboWan VAE**: Loads Wan2.1 VAE (auto-loads if not connected)
3. **Load TurboWan T5 Encoder**: Loads umT5-XXL text encoder (auto-loads if not connected)
4. **TurboDiffusion I2V**: Main image-to-video generation node
5. **Save Video**: Export frames as MP4/GIF/WebM

### Quick Start Example

#### Single-Stage Workflow (Simple)

1. Add "Load TurboWan Model" node
   - Pick any TurboWan model from dropdown (high or low noise)
   - VAE and T5 encoder auto-load if not connected

2. Add "TurboDiffusion I2V" node
   - Connect model from Model Loader
   - Connect input IMAGE
   - Optionally add a text prompt
   - Adjust num_frames (default: 77 frames)
   - Set seed for reproducibility

3. Add "Preview Image" or "Save Video" node
   - Connect frames output
   - Choose format (mp4, gif, webm)

#### Dual-Expert Workflow (Best Quality)

For highest quality, use the two-stage dual-expert approach:

1. **Shared Components** (left):
   - Add "Load TurboWan VAE" → select `Wan2.1_VAE.pth`
   - Add "Load TurboWan T5 Encoder" → select `models_t5_umt5-xxl-enc-bf16.pth`
   - Add "Load Image" for your input

2. **Stage 1: High Noise** (middle):
   - Add "Load TurboWan Model" → select high noise model (e.g., `TurboWan2.2-I2V-A14B-high-720P-quant.pth`)
   - Connect VAE and T5 encoder to model loader
   - Add "TurboDiffusion I2V" node with `boundary=0.9` (handles coarse motion)
   - Connect input image

3. **Stage 2: Low Noise** (right):
   - Add "Load TurboWan Model" → select low noise model (e.g., `TurboWan2.2-I2V-A14B-low-720P-quant.pth`)
   - Connect same VAE and T5 encoder
   - Add "TurboDiffusion I2V" node with `boundary=0.1` (refines details)
   - Connect output from Stage 1 as input

4. **Output**:
   - Add "Preview Image" to see results
   - Add "Save Video" to export final video

See [turbodiffusion_workflow_native.json](turbodiffusion_workflow_native.json) for a complete dual-expert workflow example.

### Node Parameters

#### Load TurboWan Model

- **model_filename**: Pick any TurboWan .pth file from dropdown
  - Works with both high noise and low noise variants
  - Automatically detects model from `ComfyUI/models/diffusion_models/`
  - Examples:
    - `TurboWan2.2-I2V-A14B-high-720P-quant.pth` (14.5GB, recommended for 24GB GPUs)
    - `TurboWan2.2-I2V-A14B-low-720P-quant.pth` (14.5GB, recommended for 24GB GPUs)
    - `TurboWan2.2-I2V-A14B-high-720P.pth` (28.6GB, for 40GB+ GPUs)
    - `TurboWan2.2-I2V-A14B-low-720P.pth` (28.6GB, for 40GB+ GPUs)

- **vae** (optional): Connect from "Load TurboWan VAE" node, or auto-loads `Wan2.1_VAE.pth`
- **text_encoder** (optional): Connect from "Load TurboWan T5 Encoder" node, or auto-loads encoder

#### Load TurboWan VAE

- **vae_filename**: Pick VAE file from dropdown (e.g., `Wan2.1_VAE.pth`)

#### Load TurboWan T5 Encoder

- **encoder_filename**: Pick T5 encoder file from dropdown (e.g., `models_t5_umt5-xxl-enc-bf16.pth`)

#### TurboDiffusion I2V Generator

**Required Parameters:**
- **model**: TurboDiffusion model (from Model Loader)
- **image**: Input image (ComfyUI IMAGE type)
- **num_steps**: Sampling steps (1-4, default: 4) - more steps = higher quality
- **seed**: Random seed for reproducibility

**Optional Parameters:**
- **prompt**: Text prompt to guide generation (multiline string)
- **num_frames**: Number of frames to generate (9-121, default: 77)
- **sigma_max**: Initial noise level (0-500, default: 200.0)
- **boundary**: Noise schedule switching point (0-1, default: 0.9)
- **attention_type**: Attention mechanism
  - `original`: Standard attention
  - `sla`: Sparse-Linear Attention (faster)
  - `sagesla`: Advanced SLA (requires spargeattn package)
- **sla_topk**: Sparse attention ratio (0-1, default: 0.1)
- **use_ode**: Use ODE sampling for sharper outputs (default: False)
- **adaptive_resolution**: Adapt to input image resolution (default: True)

#### Save TurboDiffusion Video

- **frames**: Frame batch from I2V Generator
- **filename_prefix**: Output filename prefix
- **fps**: Frames per second (default: 24)
- **format**: Output format (mp4, gif, webm)

## Model Information

### TurboWan2.2-I2V-A14B Model

This node uses the TurboWan2.2-I2V-A14B model from HuggingFace:
- Repository: [TurboDiffusion/TurboWan2.2-I2V-A14B-720P](https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P)
- Architecture: Wan2.2 with 14B parameters
- Type: Image-to-Video generation
- License: Apache 2.0

### Downloaded Files

On first use, the node will download:
1. Main model checkpoint (~14.5GB-28.6GB depending on variant)
2. Wan2.1 VAE checkpoint
3. umT5-XXL text encoder

Files are cached in `checkpoints/` directory for reuse.

## Performance Tips

### For 24GB GPUs (RTX 4090/5090):
- Use quantized models (`A14B-high-quant` or `A14B-low-quant`)
- Start with 480p resolution
- Reduce num_frames if experiencing OOM errors
- Use `sla` attention type for speed boost

### For 40GB+ GPUs (H100/A100):
- Use full precision models for best quality
- 720p resolution fully supported
- Can generate longer videos (more frames)

### Speed Optimization:
- Fewer num_steps = faster generation (1 step = fastest)
- Use `sla` or `sagesla` attention types
- Lower resolution (480p vs 720p)

## Troubleshooting

### CUDA Out of Memory Error
- Switch to quantized model variant
- Reduce num_frames parameter
- Use 480p instead of 720p
- Close other GPU applications
- Reduce num_steps

### Model Download Fails
- Check internet connection
- Ensure HuggingFace is accessible
- Manually download from [HuggingFace](https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P)
- Place files in `checkpoints/` directory
- Set checkpoint_path in Model Loader

### Import Error: No module named 'turbodiffusion'
- Ensure dependencies are installed: `uv sync` or `pip install -e .`
- Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
- Install turbodiffusion: `pip install turbodiffusion --no-build-isolation`

### Video Quality Issues
- Increase num_steps (2-4 recommended)
- Use full precision model instead of quantized
- Adjust sigma_max parameter
- Try use_ode=True for sharper results

## Examples

### Basic I2V Generation
```
Image → TurboWan2 Model Loader → TurboDiffusion I2V → Preview Image
                 ↓
            (A14B-high-quant, 720p)
```

### Prompted I2V with Video Export
```
Image ──┐
        ├→ TurboDiffusion I2V → Save Video (MP4)
Model ──┤
        └→ prompt: "smooth camera motion, cinematic"
```

## Credits

- **TurboDiffusion**: [thu-ml/TurboDiffusion](https://github.com/thu-ml/TurboDiffusion)
- **Model**: [TurboDiffusion/TurboWan2.2-I2V-A14B-720P](https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P)
- **ComfyUI**: [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Links

- [TurboDiffusion GitHub](https://github.com/thu-ml/TurboDiffusion)
- [TurboDiffusion HuggingFace](https://huggingface.co/TurboDiffusion)
- [ComfyUI Documentation](https://docs.comfy.org/)

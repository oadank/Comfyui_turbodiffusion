# ComfyUI TurboDiffusion I2V Node

ComfyUI-native integration for [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) Image-to-Video (I2V) generation. This node pack integrates seamlessly with ComfyUI's standard nodes for high-quality, fast video generation.

## Features

- **Fully ComfyUI Native**: Uses standard UNETLoader, CLIPLoader, VAELoader, CLIPTextEncode, KSamplerAdvanced
- **Dual-Expert Sampling**: High noise model for coarse motion + low noise model for detail refinement
- **Fast Generation**: 100-205× acceleration using TurboDiffusion framework
- **High Quality**: Supports 480p and 720p video generation up to 241 frames
- **Flexible Models**: Works with quantized (fp8) and full precision models
- **Standard Workflow**: Same pattern as official Wan workflows

## Requirements

### GPU Requirements
- **Minimum**: NVIDIA RTX 4090 (24GB VRAM) with fp8 quantized models
- **Recommended**: NVIDIA RTX 5090, H100, or A100 (40GB+ VRAM)
- Higher resolutions and more frames require more VRAM

### Software Requirements
- Python >= 3.9
- PyTorch >= 2.7.0
- ComfyUI (latest version)
- CUDA-capable GPU

## Installation

### Quick Install

1. Navigate to your ComfyUI custom_nodes directory:
```bash
cd path/to/ComfyUI/custom_nodes/
```

2. Clone this repository:
```bash
git clone https://github.com/your-org/comfyui-turbodiffusion.git
```

3. Restart ComfyUI to load the nodes

### Download Required Models

Place the following files in `ComfyUI/models/`:

**Diffusion Models** (`diffusion_models/`):
- `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` (High noise expert, ~14.5GB)
- `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` (Low noise expert, ~14.5GB)

**Text Encoder** (`text_encoders/`):
- `nsfw_wan_umt5-xxl_fp8_scaled.safetensors` (umT5-XXL, ~2GB)

**VAE** (`vae/`):
- `wan_2.1_vae.safetensors` (Wan2.1 VAE, ~2GB)

Download from: https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged

## Usage

### Workflow Overview

The TurboWan workflow follows the standard ComfyUI pattern:

```
Step 1: Load Models
├─ UNETLoader → High Noise Model
├─ UNETLoader → Low Noise Model
├─ CLIPLoader → umT5 Text Encoder
└─ VAELoader → Wan2.1 VAE

Step 2: Create Prompts
├─ CLIPTextEncode → Positive Prompt
└─ CLIPTextEncode → Negative Prompt

Step 3: Prepare I2V
└─ TurboWan I2V Sampler → Conditioning + Latent
   ├─ Input: Positive/Negative Conditioning
   ├─ Input: VAE
   ├─ Input: Start Image
   └─ Output: Modified Conditioning + Initial Latent

Step 4: Dual-Expert Sampling
├─ ModelSamplingSD3 → High Noise Config
├─ KSamplerAdvanced → Stage 1 (High Noise, steps 0-2)
├─ ModelSamplingSD3 → Low Noise Config
└─ KSamplerAdvanced → Stage 2 (Low Noise, steps 2-4)

Step 5: Decode & Save
├─ VAEDecode → Convert Latent to Images
├─ PreviewImage → Preview Video Frames
└─ Save Video → Export as MP4/GIF/WebM
```

### Example Workflow

See [turbowan_workflow.json](turbowan_workflow.json) for a complete example.

### Node Reference

#### TurboWan I2V Sampler

Prepares conditioning and latents for TurboWan I2V generation.

**Inputs:**
- `positive` (CONDITIONING): From CLIPTextEncode (positive prompt)
- `negative` (CONDITIONING): From CLIPTextEncode (negative prompt)
- `vae` (VAE): From VAELoader
- `width` (INT): Video width in pixels (default: 480)
- `height` (INT): Video height in pixels (default: 480)
- `length` (INT): Number of frames (default: 121, range: 9-241)
- `batch_size` (INT): Batch size (default: 1)
- `start_image` (IMAGE, optional): Starting image for I2V
- `clip_vision_output` (CLIP_VISION_OUTPUT, optional): CLIP vision conditioning

**Outputs:**
- `positive` (CONDITIONING): Modified positive conditioning for samplers
- `negative` (CONDITIONING): Modified negative conditioning for samplers
- `latent` (LATENT): Initial latent for KSamplerAdvanced

**Usage:**
Connect this node between CLIPTextEncode and KSamplerAdvanced. It prepares video-specific conditioning and creates the initial latent tensor.

#### Save Video

Exports frame sequences as video files.

**Inputs:**
- `frames` (IMAGE): Frame sequence from VAEDecode
- `filename_prefix` (STRING): Output filename prefix
- `fps` (INT): Frames per second (default: 16)
- `format` (COMBO): Output format (mp4, gif, webm)
- `crf` (INT): Compression quality for mp4 (default: 19)
- `save_output` (BOOLEAN): Whether to save file (default: true)

**Outputs:** None (saves to `ComfyUI/output/`)

### Typical Parameters

**For Dual-Expert Sampling:**

**Stage 1 (High Noise):**
- Steps: 4
- Start step: 0
- End step: 2
- Add noise: enable
- Return with noise: enable

**Stage 2 (Low Noise):**
- Steps: 4
- Start step: 2
- End step: 4
- Add noise: disable
- Return with noise: disable

**Recommended Settings:**
- CFG: 1.0
- Sampler: euler
- Scheduler: bong_tangent
- Model sampling shift: 8

## Model Information

### TurboWan2.2-I2V Models

**High Noise Expert:**
- Handles coarse motion and overall structure
- Processes early denoising steps
- File: `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`

**Low Noise Expert:**
- Refines details and textures
- Processes late denoising steps
- File: `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`

### Model Types

- **fp8_e4m3fn (Quantized)**: 14.5GB per model, works on 24GB GPUs
- **Full Precision**: 28.6GB per model, requires 40GB+ VRAM

### Supported Resolutions

- 480p (480x480): Lower VRAM usage
- 720p (720x720): Higher quality, more VRAM

### Frame Lengths

- Minimum: 9 frames
- Default: 121 frames (~7.5 seconds at 16fps)
- Maximum: 241 frames (~15 seconds at 16fps)

## Comparison with Official Wan Workflows

This implementation follows the exact same pattern as official Wan workflows:

| Component | Official Wan | TurboWan (This Node) |
|-----------|-------------|---------------------|
| Model Loader | UNETLoader | UNETLoader ✓ |
| Text Encoder | CLIPLoader | CLIPLoader ✓ |
| VAE | VAELoader | VAELoader ✓ |
| Prompts | CLIPTextEncode | CLIPTextEncode ✓ |
| I2V Prep | WanImageToVideo | TurboWanSampler |
| Sampling | KSamplerAdvanced | KSamplerAdvanced ✓ |
| Decode | VAEDecode | VAEDecode ✓ |

Only difference: `TurboWanSampler` replaces `WanImageToVideo` for TurboDiffusion-specific conditioning.

## Troubleshooting

### Out of Memory (OOM) Errors

- Use fp8 quantized models instead of full precision
- Reduce resolution (use 480p instead of 720p)
- Reduce number of frames
- Reduce batch size to 1

### Models Not Found

- Verify files are in correct ComfyUI folders:
  - Diffusion models: `ComfyUI/models/diffusion_models/`
  - Text encoder: `ComfyUI/models/text_encoders/`
  - VAE: `ComfyUI/models/vae/`
- Restart ComfyUI after adding models
- Check file names match exactly (case-sensitive)

### Slow Generation

- Use fp8 quantized models
- Ensure CUDA is enabled
- Check GPU utilization (should be near 100%)
- Reduce number of frames

### Poor Quality Output

- Use dual-expert sampling (high + low noise)
- Increase CFG (try 1.0-2.0)
- Use better prompts (be specific about motion)
- Try different schedulers

## Examples

### Basic I2V Generation

```
Positive Prompt: "smooth camera pan to the right, cinematic, high quality"
Negative Prompt: "static, blurry, low quality, distorted"
Resolution: 480x480
Frames: 121
Steps: 4 (dual-expert: 0-2 high, 2-4 low)
```

### High Quality Generation

```
Positive Prompt: "slow motion camera zoom in, professional cinematography, sharp details"
Negative Prompt: "static, motion blur, compression artifacts, low resolution"
Resolution: 720x720
Frames: 121
Steps: 4 (dual-expert)
CFG: 1.0
```

## Performance

- **Single frame time**: ~0.1-0.3 seconds (fp8 on RTX 4090)
- **121 frames**: ~15-30 seconds total
- **Speedup vs original**: 100-205× faster than standard diffusion

## License

Apache 2.0

## Credits

- [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) by THU Machine Learning Group
- [Wan2.2 Models](https://huggingface.co/Wan-AI) by Wan-AI Team
- ComfyUI by comfyanonymous

## Contributing

Issues and pull requests welcome at https://github.com/your-org/comfyui-turbodiffusion

## References

- TurboDiffusion Paper: https://arxiv.org/abs/2412.13631
- Wan2.2 Release: https://huggingface.co/Wan-AI
- ComfyUI: https://github.com/comfyanonymous/ComfyUI

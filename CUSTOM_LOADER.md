# Custom TurboWan Model Loader

## Problem

The TurboDiffusion quantized `.pth` models use a custom int8 quantization format that ComfyUI's standard `UNETLoader` doesn't recognize.

The models have keys like:
```
blocks.0.ffn.0.int8_weight  (quantized weights in int8)
blocks.0.ffn.0.scale        (scaling factors)
blocks.0.ffn.0.bias         (biases)
```

Instead of the standard:
```
blocks.0.ffn.0.weight  (float weights)
blocks.0.ffn.0.bias    (biases)
```

## Solution: TurboWanModelLoader

Created a custom node that:
1. Loads the `.pth` checkpoint
2. Detects if it's quantized (has `int8_weight` keys)
3. Dequantizes on-the-fly: `weight = int8_weight.float() * scale`
4. Passes the dequantized model to ComfyUI's standard loader

## Usage

In your workflow, use `TurboWanModelLoader` instead of `UNETLoader`:

```json
{
  "type": "TurboWanModelLoader",
  "widgets_values": ["TurboWan2.2-I2V-A14B-high-720P-quant.pth"]
}
```

The node will:
- ✅ Auto-detect quantized format
- ✅ Dequantize to float32 automatically
- ✅ Load into ComfyUI's standard model format
- ✅ Work with all standard ComfyUI sampling nodes

## Nodes Available

1. **TurboWanModelLoader** - Load quantized TurboDiffusion models
2. **TurboWanSampler** - Prepare I2V conditioning
3. **TurboDiffusionSaveVideo** - Export video files

## Updated Workflow

The workflow has been updated to use the custom loader:
- Node 1: `TurboWanModelLoader` (high noise model)
- Node 2: `TurboWanModelLoader` (low noise model)
- Rest of the workflow remains the same

## Memory Usage

The dequantization happens in memory:
- **Original**: ~14GB quantized on disk
- **In Memory**: ~28GB after dequantization to FP32
- **On GPU**: ComfyUI manages VRAM as usual

If you have memory constraints, consider:
- Using the official FP8 safetensors models instead
- Reducing frame count (121 → 81)
- Reducing resolution (720 → 480)

## Next Steps

1. **Restart ComfyUI** - The new node needs to be loaded
2. **Load the updated workflow** - `turbowan_workflow.json`
3. **The model will dequantize automatically** - First load may take a moment
4. **Generate!** - Everything else works the same

The custom loader handles all the complexity behind the scenes!

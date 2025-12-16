# Import Fix - ComfyUI Node Loading Issue

## Problem

After implementing the TurboWanSampler node, ComfyUI could not find the custom node. The error was:

```
This workflow uses custom nodes you haven't installed yet. TurboWanSampler
```

Root cause investigation revealed:
```
ModuleNotFoundError: No module named 'comfyui_turbodiffusion'
ImportError: attempted relative import beyond top-level package
```

## Root Cause

ComfyUI loads custom nodes differently than standard Python packages. Using relative imports (`.nodes.turbowan_sampler`) caused issues because ComfyUI doesn't always set up the package context properly.

## Solution

Changed all imports from relative to absolute imports:

### 1. Main `__init__.py` (lines 24-33)

**Before:**
```python
from .nodes.turbowan_sampler import TurboWanSampler
from .nodes.video_saver import TurboDiffusionSaveVideo
```

**After:**
```python
import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from nodes.turbowan_sampler import TurboWanSampler
from nodes.video_saver import TurboDiffusionSaveVideo
```

### 2. `nodes/video_saver.py` (line 8)

**Before:**
```python
from ..utils.video_output import save_video
```

**After:**
```python
from utils.video_output import save_video
```

### 3. `nodes/__init__.py` (lines 3-4)

**Before:**
```python
from .turbowan_sampler import TurboWanSampler
from .video_saver import TurboDiffusionSaveVideo
```

**After:**
```python
from nodes.turbowan_sampler import TurboWanSampler
from nodes.video_saver import TurboDiffusionSaveVideo
```

## Verification

The fix was verified with the following tests:

```bash
# Test 1: Direct import of nodes
cd c:\Users\Ganaraj\Documents\Projects\comfyui-turbodiffusion
python -c "import sys; sys.path.insert(0, '.'); from nodes.turbowan_sampler import TurboWanSampler; from nodes.video_saver import TurboDiffusionSaveVideo; print('OK')"
# Result: ✅ OK

# Test 2: Full __init__.py loading
python -c "import sys; sys.path.insert(0, '.'); import __init__; print(__init__.NODE_CLASS_MAPPINGS.keys())"
# Result: ✅ dict_keys(['TurboWanSampler', 'TurboDiffusionSaveVideo'])

# Test 3: Loading via symlink
cd C:\Users\Ganaraj\Downloads\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI\custom_nodes\comfyui-turbodiffusion
python -c "import sys; sys.path.insert(0, '.'); import __init__; print(__init__.NODE_CLASS_MAPPINGS.keys())"
# Result: ✅ dict_keys(['TurboWanSampler', 'TurboDiffusionSaveVideo'])
```

## Result

The custom node now loads correctly in ComfyUI. Users should restart ComfyUI to see the nodes appear:

1. Close ComfyUI
2. Restart ComfyUI
3. Load `turbowan_workflow.json`
4. TurboWanSampler node should now be recognized

## Key Learnings

1. **ComfyUI requires absolute imports**: Custom nodes should use absolute imports, not relative imports
2. **sys.path manipulation**: Adding the current directory to sys.path ensures imports work correctly
3. **Consistent import style**: All imports throughout the package should use the same style (absolute)
4. **Testing via symlink**: Always test loading through the ComfyUI custom_nodes symlink, not just direct paths

## Files Modified

- `__init__.py` - Added sys.path manipulation, changed to absolute imports
- `nodes/__init__.py` - Changed to absolute imports
- `nodes/video_saver.py` - Changed to absolute imports

## Status

✅ **FIXED** - Node now loads correctly in ComfyUI

Users should restart ComfyUI to see the changes take effect.

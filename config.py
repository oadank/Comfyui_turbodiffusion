"""
Configuration file for ComfyUI TurboDiffusion node.

Customize these settings to match your environment and preferences.
"""

from pathlib import Path

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Default model variant to use
DEFAULT_MODEL_VARIANT = "A14B-high-quant"  # Options: A14B-high, A14B-high-quant, A14B-low, A14B-low-quant

# Default resolution
DEFAULT_RESOLUTION = "720p"  # Options: 480p, 720p

# Auto-download models from HuggingFace
AUTO_DOWNLOAD_MODELS = True

# =============================================================================
# PATHS
# =============================================================================

# Checkpoint directory (None = use default: ./checkpoints)
CHECKPOINT_DIR = None  # Or set to: Path("/path/to/your/models")

# Output directory for videos (None = use ComfyUI's output dir)
OUTPUT_DIR = None  # Or set to: Path("/path/to/your/outputs")

# =============================================================================
# GENERATION DEFAULTS
# =============================================================================

# Default number of sampling steps (1-4)
DEFAULT_NUM_STEPS = 4  # Higher = better quality

# Default number of frames (9-121)
DEFAULT_NUM_FRAMES = 77  # 77 frames ≈ 3.2s at 24fps

# Default sigma_max (0-500)
DEFAULT_SIGMA_MAX = 200.0

# Default boundary for noise schedule (0-1)
DEFAULT_BOUNDARY = 0.9

# Default attention type
DEFAULT_ATTENTION_TYPE = "original"  # Options: original, sla, sagesla

# Default SLA topk ratio (0-1)
DEFAULT_SLA_TOPK = 0.1

# Use ODE sampling by default
DEFAULT_USE_ODE = False

# Adaptive resolution by default
DEFAULT_ADAPTIVE_RESOLUTION = True

# =============================================================================
# VIDEO EXPORT DEFAULTS
# =============================================================================

# Default FPS for video export
DEFAULT_FPS = 24

# Default video format
DEFAULT_VIDEO_FORMAT = "mp4"  # Options: mp4, gif, webm

# Default video quality (1-10)
DEFAULT_VIDEO_QUALITY = 8

# Optimize GIF by default
DEFAULT_OPTIMIZE_GIF = True

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# Clear CUDA cache after generation
CLEAR_CUDA_CACHE_AFTER_GENERATION = True

# Device to use (auto-detect if None)
DEVICE = None  # Options: None (auto), "cuda", "cuda:0", "cpu"

# Enable memory profiling (for debugging)
ENABLE_MEMORY_PROFILING = False

# =============================================================================
# HUGGINGFACE SETTINGS
# =============================================================================

# HuggingFace model repositories
MODEL_REPO_ID = "TurboDiffusion/TurboWan2.2-I2V-A14B-720P"
VAE_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B"
T5_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B"

# Resume downloads if interrupted
RESUME_DOWNLOADS = True

# Use symlinks for HuggingFace cache (Linux/Mac only)
USE_SYMLINKS = False  # Set to False on Windows

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

# Print detailed logs
VERBOSE_LOGGING = True

# Print model loading progress
SHOW_LOADING_PROGRESS = True

# Print generation progress
SHOW_GENERATION_PROGRESS = True

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

# Validate inputs strictly
STRICT_INPUT_VALIDATION = True

# Maximum allowed num_frames
MAX_NUM_FRAMES = 121

# Minimum allowed num_frames
MIN_NUM_FRAMES = 9

# =============================================================================
# ADVANCED SETTINGS (Don't change unless you know what you're doing)
# =============================================================================

# Model file configurations
MODEL_CONFIGS = {
    "A14B-high-720p": {
        "repo": MODEL_REPO_ID,
        "filename": "TurboWan2.2-I2V-A14B-high-720P.pth",
        "size": "28.6GB",
        "quantized": False,
    },
    "A14B-high-720p-quant": {
        "repo": MODEL_REPO_ID,
        "filename": "TurboWan2.2-I2V-A14B-high-720P-quant.pth",
        "size": "14.5GB",
        "quantized": True,
    },
    "A14B-low-720p": {
        "repo": MODEL_REPO_ID,
        "filename": "TurboWan2.2-I2V-A14B-low-720P.pth",
        "size": "28.6GB",
        "quantized": False,
    },
    "A14B-low-720p-quant": {
        "repo": MODEL_REPO_ID,
        "filename": "TurboWan2.2-I2V-A14B-low-720P-quant.pth",
        "size": "14.5GB",
        "quantized": True,
    },
}

VAE_CONFIG = {
    "repo": VAE_REPO_ID,
    "filename": "Wan2.1_VAE.pth",
}

T5_CONFIG = {
    "repo": T5_REPO_ID,
    "filename": "models_t5_umt5-xxl-enc-bf16.pth",
}

# Resolution mappings (width, height)
RESOLUTION_MAP = {
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_checkpoint_dir() -> Path:
    """Get the checkpoint directory path."""
    if CHECKPOINT_DIR is not None:
        return Path(CHECKPOINT_DIR)
    # Default: ./checkpoints
    return Path(__file__).parent / "checkpoints"


def get_output_dir() -> Path:
    """Get the output directory path."""
    if OUTPUT_DIR is not None:
        return Path(OUTPUT_DIR)
    # Default: try ComfyUI output dir, fallback to ./output
    try:
        import folder_paths
        return Path(folder_paths.get_output_directory()) / "turbodiffusion_videos"
    except ImportError:
        return Path(__file__).parent / "output"


def get_device() -> str:
    """Get the device to use for inference."""
    if DEVICE is not None:
        return DEVICE

    # Auto-detect
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    return "cpu"


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config():
    """Validate configuration settings."""
    errors = []

    # Validate model variant
    valid_variants = ["A14B-high", "A14B-high-quant", "A14B-low", "A14B-low-quant"]
    if DEFAULT_MODEL_VARIANT not in valid_variants:
        errors.append(f"Invalid DEFAULT_MODEL_VARIANT: {DEFAULT_MODEL_VARIANT}. Must be one of {valid_variants}")

    # Validate resolution
    valid_resolutions = ["480p", "720p"]
    if DEFAULT_RESOLUTION not in valid_resolutions:
        errors.append(f"Invalid DEFAULT_RESOLUTION: {DEFAULT_RESOLUTION}. Must be one of {valid_resolutions}")

    # Validate num_steps
    if not (1 <= DEFAULT_NUM_STEPS <= 4):
        errors.append(f"Invalid DEFAULT_NUM_STEPS: {DEFAULT_NUM_STEPS}. Must be between 1 and 4")

    # Validate num_frames
    if not (MIN_NUM_FRAMES <= DEFAULT_NUM_FRAMES <= MAX_NUM_FRAMES):
        errors.append(f"Invalid DEFAULT_NUM_FRAMES: {DEFAULT_NUM_FRAMES}. Must be between {MIN_NUM_FRAMES} and {MAX_NUM_FRAMES}")

    # Validate attention type
    valid_attention = ["original", "sla", "sagesla"]
    if DEFAULT_ATTENTION_TYPE not in valid_attention:
        errors.append(f"Invalid DEFAULT_ATTENTION_TYPE: {DEFAULT_ATTENTION_TYPE}. Must be one of {valid_attention}")

    # Validate video format
    valid_formats = ["mp4", "gif", "webm"]
    if DEFAULT_VIDEO_FORMAT not in valid_formats:
        errors.append(f"Invalid DEFAULT_VIDEO_FORMAT: {DEFAULT_VIDEO_FORMAT}. Must be one of {valid_formats}")

    # Validate FPS
    if not (1 <= DEFAULT_FPS <= 60):
        errors.append(f"Invalid DEFAULT_FPS: {DEFAULT_FPS}. Must be between 1 and 60")

    # Validate quality
    if not (1 <= DEFAULT_VIDEO_QUALITY <= 10):
        errors.append(f"Invalid DEFAULT_VIDEO_QUALITY: {DEFAULT_VIDEO_QUALITY}. Must be between 1 and 10")

    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Invalid configuration settings")

    return True


# Validate config on import
try:
    validate_config()
except ValueError as e:
    print(f"Configuration validation failed: {e}")
    print("Please fix config.py before using the node.")

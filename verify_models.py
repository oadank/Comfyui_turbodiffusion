"""
Quick script to verify downloaded model files are in the correct location.
Run this after downloading models manually.
"""

import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("TurboWan Model Files Verification")
print("=" * 70)

# Get script directory and checkpoint path
script_dir = Path(__file__).parent
checkpoint_dir = script_dir / "checkpoints"

print(f"\nChecking directory: {checkpoint_dir}")
print()

# Expected files and their approximate sizes
expected_files = {
    "High Noise Models": [
        ("TurboWan2.2-I2V-A14B-high-720P-quant.pth", 14.5, "GB", "quantized"),
        ("TurboWan2.2-I2V-A14B-high-720P.pth", 28.6, "GB", "full precision"),
    ],
    "Low Noise Models": [
        ("TurboWan2.2-I2V-A14B-low-720P-quant.pth", 14.5, "GB", "quantized"),
        ("TurboWan2.2-I2V-A14B-low-720P.pth", 28.6, "GB", "full precision"),
    ],
    "Required Components": [
        ("Wan2.1_VAE.pth", 2.5, "GB", "required"),
        ("models_t5_umt5-xxl-enc-bf16.pth", 2.5, "GB", "required"),
    ],
}

def format_size(size_bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

# Check what files exist
found_files = {}
missing_files = []
all_files = []

for category, files in expected_files.items():
    print(f"{category}:")
    for filename, expected_size, unit, file_type in files:
        file_path = checkpoint_dir / filename
        all_files.append((filename, file_type))

        if file_path.exists():
            size = file_path.stat().st_size
            size_str = format_size(size)
            size_gb = size / (1024**3)

            # Check if size is reasonable (within 20% of expected)
            size_ok = abs(size_gb - expected_size) < (expected_size * 0.2)

            if size_ok:
                print(f"   ✓ {filename}")
                print(f"     Size: {size_str} ({file_type})")
                found_files[filename] = file_type
            else:
                print(f"   ⚠ {filename} - SIZE MISMATCH!")
                print(f"     Expected: ~{expected_size} {unit}")
                print(f"     Found: {size_str}")
                print(f"     File may be corrupted or incomplete")
        else:
            print(f"   ✗ {filename} - NOT FOUND")
            missing_files.append((filename, file_type))
    print()

# Summary
print("=" * 70)
print("Summary")
print("=" * 70)

# Check for high noise model
has_high_noise = any(
    f in found_files
    for f in ["TurboWan2.2-I2V-A14B-high-720P-quant.pth", "TurboWan2.2-I2V-A14B-high-720P.pth"]
)

# Check for low noise model
has_low_noise = any(
    f in found_files
    for f in ["TurboWan2.2-I2V-A14B-low-720P-quant.pth", "TurboWan2.2-I2V-A14B-low-720P.pth"]
)

# Check for required components
has_vae = "Wan2.1_VAE.pth" in found_files
has_t5 = "models_t5_umt5-xxl-enc-bf16.pth" in found_files

print(f"\nFound {len(found_files)} file(s):")
for filename, file_type in found_files.items():
    print(f"  ✓ {filename} ({file_type})")

if missing_files:
    print(f"\nMissing {len(missing_files)} file(s):")
    for filename, file_type in missing_files:
        print(f"  ✗ {filename} ({file_type})")

print("\n" + "-" * 70)
print("Status:")
print("-" * 70)

# Determine what's ready
if has_high_noise and has_low_noise and has_vae and has_t5:
    print("\n✓✓✓ COMPLETE SETUP ✓✓✓")
    print("All required files are present!")
    print("\nYou have both high and low noise models.")
    print("The node will work with either variant.")
    print("\nNext steps:")
    print("1. Restart ComfyUI")
    print("2. Load TurboWan2.2 Model Loader node")
    print("3. Select your preferred variant")
    print("4. Start generating videos!")

elif (has_high_noise or has_low_noise) and has_vae and has_t5:
    print("\n✓ MINIMAL SETUP READY")
    print("You have enough files to use the node!")

    if has_high_noise:
        print("\nAvailable: High noise model + VAE + T5")
        print("You can use A14B-high variant")
    if has_low_noise:
        print("\nAvailable: Low noise model + VAE + T5")
        print("You can use A14B-low variant")

    print("\nNext steps:")
    print("1. Restart ComfyUI")
    print("2. Load TurboWan2.2 Model Loader node")
    if has_high_noise:
        print("3. Select 'A14B-high' or 'A14B-high-quant' variant")
    if has_low_noise:
        print("3. Select 'A14B-low' or 'A14B-low-quant' variant")

elif not has_vae or not has_t5:
    print("\n✗ INCOMPLETE - Missing Required Components")
    if not has_vae:
        print("  ✗ Wan2.1_VAE.pth is required")
    if not has_t5:
        print("  ✗ models_t5_umt5-xxl-enc-bf16.pth is required")

    print("\nDownload from:")
    print("https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B")

else:
    print("\n✗ NO MODELS FOUND")
    print("\nYou need to download:")
    print("1. At least one TurboWan2.2 model (high OR low, quantized or full)")
    print("2. Wan2.1_VAE.pth")
    print("3. models_t5_umt5-xxl-enc-bf16.pth")
    print("\nSee DOWNLOAD_MODELS.md for download links")

print("\n" + "=" * 70)

# Disk space check
print("\nDisk Space:")
print("-" * 70)
try:
    import shutil
    total, used, free = shutil.disk_usage(checkpoint_dir)
    print(f"Available: {format_size(free)}")

    if free < 35 * (1024**3):  # Less than 35GB free
        print("⚠ Warning: Low disk space!")
        print("  You need ~35GB free for quantized models")
        print("  Or ~65GB free for full precision models")
except:
    print("Could not check disk space")

print("=" * 70)

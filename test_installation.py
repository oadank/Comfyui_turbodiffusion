"""
Test script to verify the ComfyUI TurboDiffusion node setup.
Run this from the ComfyUI directory to verify everything is ready.
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("ComfyUI TurboDiffusion Installation Test")
print("=" * 70)

# Test 1: Check if symlink exists
print("\n1. Checking symlink...")
symlink_path = Path("custom_nodes/comfyui-turbodiffusion")
if symlink_path.exists():
    print(f"   ✓ Symlink exists: {symlink_path.resolve()}")
else:
    print(f"   ✗ Symlink NOT found at: {symlink_path}")
    sys.exit(1)

# Test 2: Check if __init__.py exists
print("\n2. Checking __init__.py...")
init_file = symlink_path / "__init__.py"
if init_file.exists():
    print(f"   ✓ __init__.py found")
else:
    print(f"   ✗ __init__.py NOT found")
    sys.exit(1)

# Test 3: Check required dependencies
print("\n3. Checking dependencies...")
required_packages = [
    ("torch", "2.7.0"),
    ("huggingface_hub", "0.20.0"),
    ("PIL", "10.0.0"),
    ("numpy", "1.24.0"),
    ("tqdm", "4.65.0"),
]

missing = []
for package, min_version in required_packages:
    try:
        if package == "PIL":
            import PIL
            pkg = PIL
        else:
            pkg = __import__(package)

        version = getattr(pkg, "__version__", "unknown")
        print(f"   ✓ {package}: {version}")
    except ImportError:
        print(f"   ✗ {package}: NOT INSTALLED")
        missing.append(package)

if missing:
    print(f"\n   Missing packages: {', '.join(missing)}")
    print("   Install with: pip install " + " ".join(missing))

# Test 4: Check optional video dependencies
print("\n4. Checking optional dependencies (video export)...")
optional = [
    ("cv2", "opencv-python"),
    ("imageio", "imageio"),
]

for import_name, package_name in optional:
    try:
        __import__(import_name)
        print(f"   ✓ {package_name}: installed")
    except ImportError:
        print(f"   ○ {package_name}: not installed (optional)")

# Test 5: Check turbodiffusion
print("\n5. Checking turbodiffusion package...")
try:
    import turbodiffusion
    version = getattr(turbodiffusion, "__version__", "unknown")
    print(f"   ✓ turbodiffusion: {version}")
    turbodiffusion_installed = True
except ImportError:
    print(f"   ✗ turbodiffusion: NOT INSTALLED (inference will not work)")
    print("   Install with: pip install turbodiffusion --no-build-isolation")
    print("   Note: May require Windows Long Path support enabled")
    turbodiffusion_installed = False

# Test 6: Check file structure
print("\n6. Checking file structure...")
required_files = [
    "nodes/__init__.py",
    "nodes/model_loader.py",
    "nodes/i2v_generator.py",
    "nodes/video_saver.py",
    "utils/__init__.py",
    "utils/model_management.py",
    "utils/preprocessing.py",
    "utils/video_output.py",
    "config.py",
    "example_workflow.json",
]

all_files_exist = True
for file_path in required_files:
    full_path = symlink_path / file_path
    if full_path.exists():
        print(f"   ✓ {file_path}")
    else:
        print(f"   ✗ {file_path} NOT FOUND")
        all_files_exist = False

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

if all_files_exist and not missing:
    print("\n✓ Node structure: READY")
    print("✓ Dependencies: INSTALLED")

    if turbodiffusion_installed:
        print("✓ TurboDiffusion: INSTALLED")
        print("\n🎉 Everything is ready! Restart ComfyUI to load the nodes.")
    else:
        print("⚠ TurboDiffusion: NOT INSTALLED")
        print("\n✓ Nodes will appear in ComfyUI, but inference won't work yet.")
        print("  Install turbodiffusion to enable actual video generation.")
        print("\n  To enable Windows Long Paths:")
        print("  1. Open Registry Editor (regedit)")
        print("  2. Go to: HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\FileSystem")
        print("  3. Set LongPathsEnabled = 1")
        print("  4. Restart computer")
        print("  5. Install: pip install turbodiffusion --no-build-isolation")
else:
    print("\n✗ Setup incomplete. Please check errors above.")
    sys.exit(1)

print("\n" + "=" * 70)
print("Next Steps:")
print("=" * 70)
print("1. Restart ComfyUI")
print("2. Look for nodes in: Add Node → video → turbodiffusion")
print("3. Load example workflow: custom_nodes/comfyui-turbodiffusion/example_workflow.json")
print("4. Test the nodes!")
print("=" * 70)

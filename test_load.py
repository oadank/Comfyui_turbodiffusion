#!/usr/bin/env python
"""Test script to verify the custom node can be loaded by ComfyUI."""

import sys
import os
import importlib.util

def test_load():
    """Test loading the custom node module."""
    print("=" * 60)
    print("Testing ComfyUI Custom Node Loading")
    print("=" * 60)

    # Simulate ComfyUI's loading process
    module_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(module_dir, '__init__.py')

    print(f"\nModule directory: {module_dir}")
    print(f"Module path: {module_path}")
    print(f"Module exists: {os.path.exists(module_path)}")

    # Check if nodes directory exists
    nodes_dir = os.path.join(module_dir, 'nodes')
    print(f"\nNodes directory: {nodes_dir}")
    print(f"Nodes exists: {os.path.exists(nodes_dir)}")

    # List node files
    if os.path.exists(nodes_dir):
        node_files = os.listdir(nodes_dir)
        print(f"Node files: {node_files}")

    # Try to load the module
    print("\n" + "=" * 60)
    print("Attempting to load module...")
    print("=" * 60)

    try:
        # Add parent directory to path
        parent_dir = os.path.dirname(module_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # Load using importlib (how ComfyUI does it)
        spec = importlib.util.spec_from_file_location('comfyui_turbodiffusion', module_path)
        if spec is None:
            print("ERROR: Could not create module spec")
            return False

        module = importlib.util.module_from_spec(spec)
        sys.modules['comfyui_turbodiffusion'] = module
        spec.loader.exec_module(module)

        print("\nSUCCESS: Module loaded!")
        print(f"Module: {module}")
        print(f"Has NODE_CLASS_MAPPINGS: {hasattr(module, 'NODE_CLASS_MAPPINGS')}")
        print(f"Has NODE_DISPLAY_NAME_MAPPINGS: {hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS')}")

        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            print(f"\nRegistered nodes: {list(module.NODE_CLASS_MAPPINGS.keys())}")
            print(f"Display names: {module.NODE_DISPLAY_NAME_MAPPINGS}")

            # Test instantiation
            print("\n" + "=" * 60)
            print("Testing node instantiation...")
            print("=" * 60)

            for node_name, node_class in module.NODE_CLASS_MAPPINGS.items():
                print(f"\nNode: {node_name}")
                print(f"  Class: {node_class}")
                print(f"  Has INPUT_TYPES: {hasattr(node_class, 'INPUT_TYPES')}")
                print(f"  Has RETURN_TYPES: {hasattr(node_class, 'RETURN_TYPES')}")
                print(f"  Has FUNCTION: {hasattr(node_class, 'FUNCTION')}")

                try:
                    # Try to get INPUT_TYPES (this is what ComfyUI does)
                    input_types = node_class.INPUT_TYPES()
                    print(f"  INPUT_TYPES: OK")
                except Exception as e:
                    print(f"  INPUT_TYPES ERROR: {e}")

        return True

    except Exception as e:
        print(f"\nERROR loading module: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_load()
    sys.exit(0 if success else 1)

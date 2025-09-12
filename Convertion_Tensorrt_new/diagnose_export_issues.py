#!/usr/bin/env python3
"""
Diagnostic script to identify issues preventing ONNX export
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available."""
    print("="*60)
    print("DEPENDENCY CHECK")
    print("="*60)
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('onnx', 'ONNX'),
        ('timm', 'Timm (for pretrained models)'),
    ]
    
    missing = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} is available")
        except ImportError as e:
            print(f"❌ {name} is missing: {e}")
            missing.append((module, name))
    
    return missing

def check_file_paths():
    """Check if required files exist."""
    print("\n" + "="*60)
    print("FILE PATH CHECK")
    print("="*60)
    
    # Check if the main model file exists
    model_file = Path(__file__).parent / "accurate_matchanything_trt_dyn.py"
    if model_file.exists():
        print(f"✅ Model file exists: {model_file}")
    else:
        print(f"❌ Model file missing: {model_file}")
    
    # Check parent directory structure
    parent_dir = Path(__file__).parent.parent / "Convertion_Tensorrt"
    if parent_dir.exists():
        print(f"✅ Parent directory exists: {parent_dir}")
        # List important files in parent directory
        important_files = ["gp_trt.py", "encoders_trt_full.py"]
        for file in important_files:
            file_path = parent_dir / file
            if file_path.exists():
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file}")
    else:
        print(f"❌ Parent directory missing: {parent_dir}")
    
    # Check if improved weight loader exists
    weight_loader = Path(__file__).parent / "improved_weight_loader.py"
    if weight_loader.exists():
        print(f"✅ Weight loader exists: {weight_loader}")
    else:
        print(f"❌ Weight loader missing: {weight_loader}")

def check_python_environment():
    """Check Python environment details."""
    print("\n" + "="*60)
    print("PYTHON ENVIRONMENT CHECK")
    print("="*60)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries

def suggest_fixes(missing_deps):
    """Suggest fixes for identified issues."""
    print("\n" + "="*60)
    print("SUGGESTED FIXES")
    print("="*60)
    
    if missing_deps:
        print("Missing dependencies detected. To fix:")
        print("1. Install PyTorch:")
        print("   pip install torch torchvision torchaudio")
        print("2. Install ONNX:")
        print("   pip install onnx")
        print("3. Install additional dependencies:")
        print("   pip install timm")
        print("4. Or install all at once:")
        print("   pip install torch torchvision torchaudio onnx timm")
    
    print("\nTo run the export script successfully:")
    print("1. Ensure you're in the correct environment with all dependencies")
    print("2. Use the correct checkpoint path (if available)")
    print("3. Make sure you have sufficient memory (the model is large)")
    print("4. Consider using smaller input dimensions (e.g., 280x280) for testing")
    
    print("\nExample command after fixing dependencies:")
    print("python3 export_fixed_weights_optimized_debug.py \\")
    print("  --onnx output/matchanything_optimized.onnx \\")
    print("  --H 280 --W 280")

def main():
    print("MatchAnything ONNX Export Diagnostic Tool")
    
    missing_deps = check_dependencies()
    check_file_paths()
    check_python_environment()
    suggest_fixes(missing_deps)
    
    print("\n" + "="*60)
    if missing_deps:
        print("❌ Issues found - please address the missing dependencies above")
        return 1
    else:
        print("✅ Environment looks good - you should be able to run the export")
        return 0

if __name__ == "__main__":
    sys.exit(main())
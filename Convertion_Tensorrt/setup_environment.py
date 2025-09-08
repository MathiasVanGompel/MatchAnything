#!/usr/bin/env python3
"""
Environment setup script for MatchAnything TensorRT conversion.
This script helps users verify their environment and install dependencies.
"""

import os
import sys
import subprocess
import importlib.util


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    else:
        print("✅ Python version is compatible")
        return True


def check_cuda():
    """Check CUDA availability"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            # Extract CUDA version if possible
            lines = result.stdout.split("\n")
            for line in lines:
                if "CUDA Version" in line:
                    cuda_version = line.split("CUDA Version: ")[1].split()[0]
                    print(f"   CUDA Version: {cuda_version}")
            return True
        else:
            print("❌ nvidia-smi not found or failed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - CUDA may not be installed")
        return False


def check_tensorrt():
    """Check TensorRT installation"""
    try:
        result = subprocess.run(["trtexec", "--help"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ TensorRT (trtexec) is available")
            return True
        else:
            print("❌ trtexec not found or failed")
            return False
    except FileNotFoundError:
        print("❌ trtexec not found - TensorRT may not be installed")
        return False


def check_python_packages():
    """Check required Python packages"""
    required_packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "onnx": "ONNX",
        "cv2": "OpenCV (cv2)",
        "numpy": "NumPy",
    }

    optional_packages = {
        "tensorrt": "TensorRT Python API",
        "pycuda": "PyCUDA",
        "onnx_graphsurgeon": "ONNX GraphSurgeon",
    }

    print("\nChecking required Python packages:")
    all_required = True

    for package, name in required_packages.items():
        if importlib.util.find_spec(package):
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - Missing")
            all_required = False

    print("\nChecking optional Python packages:")
    for package, name in optional_packages.items():
        if importlib.util.find_spec(package):
            print(f"✅ {name}")
        else:
            print(f"⚠️  {name} - Missing (optional for full TensorRT functionality)")

    return all_required


def check_roma_installation():
    """Check if ROMA is properly installed"""
    current_dir = os.path.dirname(__file__)
    roma_root = os.path.join(
        os.path.dirname(current_dir),
        "imcui",
        "third_party",
        "MatchAnything",
        "third_party",
        "ROMA",
    )

    print("\nChecking ROMA installation:")
    print(f"Expected ROMA path: {roma_root}")

    if os.path.exists(roma_root):
        print("✅ ROMA directory exists")

        # Check for key files
        key_files = ["roma/models/transformer/dinov2.py", "roma/__init__.py"]

        all_files_exist = True
        for file_path in key_files:
            full_path = os.path.join(roma_root, file_path)
            if os.path.exists(full_path):
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} - Missing")
                all_files_exist = False

        return all_files_exist
    else:
        print("❌ ROMA directory not found")
        print("   Please ensure the ROMA submodule is properly initialized")
        return False


def print_installation_instructions():
    """Print installation instructions"""
    print("\n" + "=" * 60)
    print("INSTALLATION INSTRUCTIONS")
    print("=" * 60)

    print("\n1. Install CUDA and TensorRT:")
    print("   - Download CUDA from: https://developer.nvidia.com/cuda-downloads")
    print("   - Download TensorRT from: https://developer.nvidia.com/tensorrt")
    print("   - Follow NVIDIA's installation guides for your platform")

    print("\n2. Install Python dependencies:")
    print(
        "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    )
    print("   pip install onnx>=1.12.0 opencv-python numpy")
    print(
        "   pip install tensorrt pycuda onnx-graphsurgeon  # Optional but recommended"
    )

    print("\n3. Initialize ROMA submodule (if not done):")
    print("   git submodule update --init --recursive")

    print("\n4. Test the setup:")
    print("   cd Convertion_Tensorrt")
    print("   python3 test_structure.py")


def main():
    print("MatchAnything TensorRT Environment Setup")
    print("=" * 50)

    # Run all checks
    python_ok = check_python_version()
    cuda_ok = check_cuda()
    tensorrt_ok = check_tensorrt()
    packages_ok = check_python_packages()
    roma_ok = check_roma_installation()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if python_ok and cuda_ok and tensorrt_ok and packages_ok and roma_ok:
        print("🎉 Environment is ready for TensorRT conversion!")
        print("\nNext steps:")
        print("1. Run: ./build_accurate_tensorrt.sh --help")
        print(
            "2. Convert your model: ./build_accurate_tensorrt.sh --ckpt /path/to/checkpoint.ckpt"
        )
    else:
        print("❌ Environment setup incomplete.")
        print("Please address the missing components above.")

        if not (python_ok and cuda_ok and tensorrt_ok and packages_ok):
            print_installation_instructions()


if __name__ == "__main__":
    main()

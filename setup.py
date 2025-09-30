#!/usr/bin/env python3
"""Setup script for the DreamPRM ATP project.

This script helps set up the development environment and validates
that all dependencies are properly installed.
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install dependencies from requirements.txt."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        "torch",
        "transformers", 
        "peft",
        "numpy",
        "tqdm"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_gpu_availability():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (count: {gpu_count})")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  No GPU detected - training will use CPU (much slower)")
            return False
    except ImportError:
        print("❌ Cannot check GPU - PyTorch not installed")
        return False

def create_sample_config():
    """Create a sample configuration file."""
    config_content = '''# Sample configuration for DreamPRM training
# Copy this file and modify as needed

# Data paths (update these to point to your actual data)
train_json_file = "data/train_sample.jsonl"
meta_json_file = "data/meta_sample.jsonl"

# Model configuration
model_name = "Qwen/Qwen2.5-Math-PRM-7B"
max_length = 4096

# Training parameters
total_steps = 2000
batch_size = 1
meta_batch_size = 1
lr = 2e-4
meta_lr = 1e-1

# LoRA configuration
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

# System settings
device = "cuda"  # or "cpu"
precision = "bf16"  # or "fp16" or "fp32"
seed = 42

# Experiment tracking
project_name = "DreamPRM-ATP"
disable_wandb = false

# Checkpointing
weights_path = "./checkpoints"
save_every_steps = 500
'''
    
    config_file = Path(__file__).parent / "config_sample.ini"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Sample configuration created: {config_file}")

def validate_project_structure():
    """Validate that the project structure is correct."""
    base_path = Path(__file__).parent
    
    required_files = [
        "main.py",
        "config.py",
        "requirements.txt",
        "data/__init__.py",
        "models/__init__.py", 
        "training/__init__.py",
        "utils/__init__.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def main():
    """Main setup function."""
    print("🚀 DreamPRM ATP Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Validate project structure
    print("\n📁 Checking project structure...")
    if not validate_project_structure():
        print("❌ Project structure is incomplete")
        return 1
    
    # Check/install dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("\n🔧 Installing missing dependencies...")
        if not install_dependencies():
            return 1
        
        # Re-check after installation
        print("\n🔍 Re-checking dependencies...")
        if not check_dependencies():
            print("❌ Some dependencies still missing after installation")
            return 1
    
    # Check GPU
    print("\n🖥️  Checking GPU availability...")
    check_gpu_availability()
    
    # Create sample config
    print("\n⚙️  Creating sample configuration...")
    create_sample_config()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Prepare your training data in JSONL format")
    print("2. Update the data paths in config_sample.ini")
    print("3. Run training: python main.py --help")
    print("4. Or try the examples: python example.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
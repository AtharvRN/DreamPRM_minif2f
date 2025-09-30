#!/usr/bin/env python3
"""
Dataset Visualizer Launcher

This script ensures all dependencies are installed and launches the visualizer.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages for the visualizer."""
    requirements_file = Path(__file__).parent / "visualizer_requirements.txt"
    
    if not requirements_file.exists():
        print("Requirements file not found!")
        return False
    
    try:
        print("Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_dependencies():
    """Check if required packages are available."""
    required_packages = ["flask", "plotly", "numpy"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def main():
    """Main launcher function."""
    print("🔍 Checking dependencies...")
    
    missing = check_dependencies()
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("📦 Installing dependencies...")
        
        if not install_requirements():
            print("❌ Failed to install dependencies. Please install manually:")
            print("pip install flask plotly numpy")
            sys.exit(1)
    else:
        print("✅ All dependencies are available!")
    
    # Import and run the visualizer
    try:
        from visualizer import main as run_visualizer
        run_visualizer()
    except Exception as e:
        print(f"❌ Error running visualizer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
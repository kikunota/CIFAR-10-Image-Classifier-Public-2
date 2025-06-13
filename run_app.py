#!/usr/bin/env python3
"""
🚀 Streamlit App Launcher

Quick launcher script for the CIFAR-10 Image Classification Web App.
Handles environment setup and launches the Streamlit application.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'torch', 'torchvision', 'plotly', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_model_exists():
    """Check if trained model exists"""
    model_path = Path("checkpoints/cifar10_simple_cnn/model_best.pth")
    
    if not model_path.exists():
        print("❌ Trained model not found!")
        print(f"   Expected: {model_path}")
        print("\n🏋️ Train a model first:")
        print("   python src/training/train_cifar10_clean.py")
        return False
    
    print("✅ Model found!")
    return True

def launch_app():
    """Launch the Streamlit application"""
    print("🚀 Launching CIFAR-10 Image Classification App...")
    print("📱 The app will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.headless", "false",
            "--server.enableCORS", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")

def main():
    """Main launcher function"""
    print("🖼️ CIFAR-10 Image Classification Web App Launcher")
    print("=" * 50)
    
    # Check current directory
    if not os.path.exists("streamlit_app.py"):
        print("❌ Please run this script from the project root directory")
        print("   containing streamlit_app.py")
        sys.exit(1)
    
    # Check requirements
    print("📋 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    print("✅ All packages installed!")
    
    # Check model
    print("🔍 Checking for trained model...")
    if not check_model_exists():
        response = input("\n❓ Continue anyway? (y/N): ").lower().strip()
        if response != 'y':
            sys.exit(1)
    
    # Launch app
    launch_app()

if __name__ == "__main__":
    main()
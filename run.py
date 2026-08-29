#!/usr/bin/env python
"""
Startup script for AI Grooming Assistant
Handles environment setup and starts the Flask app
"""

import os
import sys
import platform
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    required_packages = ['flask', 'torch', 'torchvision', 'transformers', 'PIL']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print(f"\nInstall with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies satisfied!\n")
    return True

def check_models():
    """Check if model files exist"""
    print("🔍 Checking model files...")
    models_dir = Path("models")
    required_models = [
        "face_shape_model.pth",
        "hairstyle_model.pth",
        "skin_type_model.pth"
    ]
    
    models_found = True
    for model in required_models:
        model_path = models_dir / model
        if model_path.exists():
            print(f"✓ {model}")
        else:
            print(f"✗ {model} (will be required at runtime)")
            models_found = False
    
    if not models_found:
        print("\n⚠️  Some model files are missing.")
        print("   Download them from the repository or they will be requested at runtime.")
    else:
        print("\n✅ All model files found!\n")
    
    return True

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    directories = [
        "static/uploads",
        "models",
        "grooming_suggestions"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ {directory}")
    
    print()

def main():
    """Main startup function"""
    print("\n" + "="*50)
    print("🧔 AI GROOMING ASSISTANT")
    print("="*50 + "\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required!")
        sys.exit(1)
    
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()}\n")
    
    # Setup directories
    setup_directories()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Check models
    check_models()
    
    # Import after dependency check
    print("🚀 Starting Flask application...\n")
    print("="*50)
    print("📱 Access the app at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

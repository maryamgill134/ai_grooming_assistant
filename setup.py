#!/usr/bin/env python
"""
Setup script for AI Grooming Assistant
Handles initial setup and configuration
"""

import os
import sys
from pathlib import Path
import subprocess

def create_directories():
    """Create necessary project directories"""
    print("\n📁 Creating directories...")
    directories = [
        "static/uploads",
        "models",
        "grooming_suggestions",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}/")

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing dependencies...")
    print("This may take a few minutes...\n")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Failed to install dependencies")
        return False

def verify_installation():
    """Verify all necessary components are in place"""
    print("\n✔️ Verifying installation...")
    
    # Check Python version
    version_ok = sys.version_info >= (3, 8)
    print(f"{'✓' if version_ok else '✗'} Python {sys.version.split()[0]}")
    
    # Check directories
    required_dirs = ["models", "static/uploads", "grooming_suggestions"]
    for directory in required_dirs:
        exists = Path(directory).exists()
        print(f"{'✓' if exists else '✗'} {directory}/")
    
    # Check key files
    required_files = ["app.py", "predict.py", "requirements.txt", "config.py"]
    for file in required_files:
        exists = Path(file).exists()
        print(f"{'✓' if exists else '✗'} {file}")
    
    # Check models
    print("\n🤖 Checking ML models:")
    models = [
        "models/face_shape_model.pth",
        "models/hairstyle_model.pth", 
        "models/skin_type_model.pth"
    ]
    for model in models:
        exists = Path(model).exists()
        status = "Found" if exists else "Missing (will download at runtime)"
        print(f"  {'✓' if exists else 'ℹ'} {model} - {status}")
    
    return True

def create_env_file():
    """Create .env file for configuration"""
    env_file = Path(".env")
    if not env_file.exists():
        print("\n🔧 Creating .env file...")
        env_content = """# AI Grooming Assistant Configuration
FLASK_ENV=development
DEBUG=True
SECRET_KEY=your-secret-key-here-change-in-production
MAX_UPLOAD_SIZE=16777216
LOG_LEVEL=INFO
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✓ .env file created")
    else:
        print("✓ .env file already exists")

def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("🧔 AI GROOMING ASSISTANT - SETUP")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    
    print(f"✓ Python {sys.version.split()[0]}")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Setup partially complete. Try running:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Create env file
    create_env_file()
    
    # Verify installation
    verify_installation()
    
    print("\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)
    print("\n🚀 To start the application, run:")
    print("   python run.py")
    print("\n📱 Then open: http://localhost:5000")
    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

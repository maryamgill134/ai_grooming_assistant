"""
Configuration file for AI Grooming Assistant
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Flask Configuration
class Config:
    """Base configuration"""
    FLASK_APP = 'app.py'
    FLASK_ENV = 'development'
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    
    # Upload Configuration
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    
    # Model Configuration
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    FACE_SHAPE_MODEL = os.path.join(MODELS_DIR, 'face_shape_model.pth')
    HAIRSTYLE_MODEL = os.path.join(MODELS_DIR, 'hairstyle_model.pth')
    SKIN_TYPE_MODEL = os.path.join(MODELS_DIR, 'skin_type_model.pth')
    GENDER_MODEL = "rizvandwiki/gender-classification-2"  # Hugging Face model
    
    # Suggestions Configuration
    SUGGESTIONS_FILE = os.path.join(BASE_DIR, 'grooming_suggestions/suggestions.json')
    
    # Device Configuration
    DEVICE = 'cuda'  # Will auto-fallback to 'cpu' if CUDA not available
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = os.path.join(BASE_DIR, 'logs/app.log')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    FLASK_ENV = 'production'

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    UPLOAD_FOLDER = '/tmp/ai_grooming_uploads'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(env=None):
    """Get configuration based on environment"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])

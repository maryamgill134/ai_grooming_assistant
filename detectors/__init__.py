"""
Detectors module - AI models for grooming attribute detection
"""

from .face_shape_model import FaceShapeDetector
from .gender_detection_model import GenderDetector
from .hair_style_model import HairStyleDetector
from .skin_type_model import SkinTypeDetector

__all__ = [
    'FaceShapeDetector',
    'GenderDetector', 
    'HairStyleDetector',
    'SkinTypeDetector'
]

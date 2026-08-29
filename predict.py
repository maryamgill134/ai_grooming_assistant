from detectors.face_shape_model import FaceShapeDetector
from detectors.gender_detection_model import GenderDetector
from detectors.hair_style_model import HairStyleDetector
from detectors.skin_type_model import SkinTypeDetector
import os
import traceback
import cv2

# Initialize detectors with error handling
face_shape_detector = None
gender_detector = None
hairstyle_detector = None
skin_type_detector = None

def initialize_models():
    """Initialize all models with proper error handling"""
    global face_shape_detector, gender_detector, hairstyle_detector, skin_type_detector
    
    try:
        print("Loading Face Shape Detector...")
        if os.path.exists("models/face_shape_model.pth"):
            face_shape_detector = FaceShapeDetector("models/face_shape_model.pth")
            print("✓ Face Shape Detector loaded")
        else:
            print("✗ Face shape model not found at models/face_shape_model.pth")
    except Exception as e:
        print(f"Error loading Face Shape Detector: {e}")
        traceback.print_exc()

    try:
        print("Loading Gender Detector...")
        gender_detector = GenderDetector("rizvandwiki/gender-classification-2")
        print("✓ Gender Detector loaded")
    except Exception as e:
        print(f"Error loading Gender Detector: {e}")
        print("Note: Gender detector requires internet connection to download model from Hugging Face")
        traceback.print_exc()

    try:
        print("Loading Hair Style Detector...")
        if os.path.exists("models/hairstyle_model.pth"):
            hairstyle_detector = HairStyleDetector("models/hairstyle_model.pth")
            print("✓ Hair Style Detector loaded")
        else:
            print("✗ Hair style model not found at models/hairstyle_model.pth")
    except Exception as e:
        print(f"Error loading Hair Style Detector: {e}")
        traceback.print_exc()

    try:
        print("Loading Skin Type Detector...")
        if os.path.exists("models/skin_type_model.pth"):
            skin_type_detector = SkinTypeDetector("models/skin_type_model.pth")
            print("✓ Skin Type Detector loaded")
        else:
            print("✗ Skin type model not found at models/skin_type_model.pth")
    except Exception as e:
        print(f"Error loading Skin Type Detector: {e}")
        traceback.print_exc()

# Load models on startup
initialize_models()

def image_has_face(image_path):
    """Return whether the uploaded image contains a detectable face."""
    image = cv2.imread(image_path)
    if image is None:
        return False

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return len(faces) > 0

def predict_attributes(image_path):
    """Predict all attributes for an image"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not image_has_face(image_path):
            raise ValueError(
                "No face detected. Please upload a clear, front-facing face photo."
            )
        
        attributes = {}
        
        # Detect Face Shape
        if face_shape_detector:
            try:
                face_shape, confidence = face_shape_detector.detect_face_shape(image_path)
                attributes["face_shape"] = face_shape
                attributes["face_shape_confidence"] = round(confidence, 3)
            except Exception as e:
                print(f"Error detecting face shape: {e}")
                attributes["face_shape"] = "Unknown"
                attributes["face_shape_confidence"] = 0.0
        
        # Detect Gender
        if gender_detector:
            try:
                gender, confidence = gender_detector.detect_gender(image_path)
                attributes["gender"] = gender if gender else "Unknown"
                attributes["gender_confidence"] = round(confidence, 3) if confidence else 0.0
            except Exception as e:
                print(f"Error detecting gender: {e}")
                attributes["gender"] = "Unknown"
                attributes["gender_confidence"] = 0.0
        
        # Detect Hair Style
        if hairstyle_detector:
            try:
                hair_type, confidence = hairstyle_detector.detect_hair_style(image_path)
                attributes["hair_type"] = hair_type
                attributes["hair_type_confidence"] = round(confidence, 3)
            except Exception as e:
                print(f"Error detecting hair style: {e}")
                attributes["hair_type"] = "Unknown"
                attributes["hair_type_confidence"] = 0.0
        
        # Detect Skin Type
        if skin_type_detector:
            try:
                skin_type, confidence = skin_type_detector.detect_skin_type(image_path)
                attributes["skin_type"] = skin_type
                attributes["skin_type_confidence"] = round(confidence, 3)
            except Exception as e:
                print(f"Error detecting skin type: {e}")
                attributes["skin_type"] = "Unknown"
                attributes["skin_type_confidence"] = 0.0
        
        if not attributes:
            raise Exception("No detectors available. Please check model files.")
        
        return attributes

    except ValueError:
        raise
    except Exception as e:
        print(f"Error in predict_attributes: {e}")
        traceback.print_exc()
        return None

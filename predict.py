# from detectors.face_shape_model import FaceShapeDetector
# from detectors.gender_detection_model import GenderDetector
# from detectors.hair_style_model import HairStyleDetector
# from detectors.skin_type_model import SkinTypeDetector

# # Load models once
# face_shape_detector = FaceShapeDetector("models/face_shape_model.pth")
# gender_detector = GenderDetector("models/gender_model")
# hairstyle_detector = HairStyleDetector("models/hairstyle_model.pth")
# skin_type_detector = SkinTypeDetector("models/skin_type_model.pth")

# def predict_attributes(image_path):
#     face_shape, _ = face_shape_detector.detect_face_shape(image_path)
#     gender, _ = gender_detector.detect_gender(image_path)
#     hair_type, _ = hairstyle_detector.detect_hair_style(image_path)
#     skin_type, _ = skin_type_detector.detect_skin_type(image_path)

#     return {
#         "face_shape": face_shape,
#         "gender": gender,
#         "hair_type": hair_type,
#         "skin_type": skin_type
#     }
from detectors.face_shape_model import FaceShapeDetector
from detectors.gender_detection_model import GenderDetector
from detectors.hair_style_model import HairStyleDetector
from detectors.skin_type_model import SkinTypeDetector

# Load models once
face_shape_detector = FaceShapeDetector("models/face_shape_model.pth")
gender_detector = GenderDetector("rizvandwiki/gender-classification-2")  # Assumes this is from huggingface or torch hub
hairstyle_detector = HairStyleDetector("models/hairstyle_model.pth")
skin_type_detector = SkinTypeDetector("models/skin_type_model.pth")

def predict_attributes(image_path):
    face_shape, _ = face_shape_detector.detect_face_shape(image_path)
    gender, _ = gender_detector.detect_gender(image_path)
    hair_type, _ = hairstyle_detector.detect_hair_style(image_path)
    skin_type, _ = skin_type_detector.detect_skin_type(image_path)

    return {
        "face_shape": face_shape,
        "gender": gender,
        "hair_type": hair_type,
        "skin_type": skin_type
    }

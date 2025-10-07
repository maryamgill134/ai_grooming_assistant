# from flask import Flask, request, render_template, jsonify
# import os
# from werkzeug.utils import secure_filename
# from predict import predict_attributes
# import json

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure folder exists

# # Load suggestions from JSON
# with open('grooming_suggestions/suggestions.json', 'r') as f:
#     suggestions = json.load(f)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     image = request.files['image']
#     if image.filename == '':
#         return jsonify({'error': 'Empty filename'}), 400

#     filename = secure_filename(image.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     image.save(filepath)

#     try:
#         attributes = predict_attributes(filepath)
#     except Exception as e:
#         return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

#     response = {
#         "attributes": attributes,
#         "suggestions": {
#             "face_shape": suggestions["face_shape"].get(attributes.get("face_shape"), "No suggestion"),
#             "gender": suggestions["gender"].get(attributes.get("gender"), "No suggestion"),
#             "hair_type": suggestions["hair_type"].get(attributes.get("hair_type"), "No suggestion"),
#             "skin_type": suggestions["skin_type"].get(attributes.get("skin_type"), "No suggestion")
#         }
#     }

#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, request, render_template, jsonify
# import os
# from werkzeug.utils import secure_filename
# from predict import predict_attributes
# import json

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure folder exists

# # Load suggestions from JSON
# with open('grooming_suggestions/suggestions.json', 'r') as f:
#     suggestions = json.load(f)

# def get_suggestion(category, key):
#     key_lower = key.lower() if key else ''
#     return suggestions.get(category, {}).get(key_lower, "No suggestion")

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     image = request.files['image']
#     if image.filename == '':
#         return jsonify({'error': 'Empty filename'}), 400

#     filename = secure_filename(image.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     image.save(filepath)

#     try:
#         attributes = predict_attributes(filepath)
#     except Exception as e:
#         return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

#     response = {
#         "attributes": attributes,
#         "suggestions": {
#             "face_shape": get_suggestion("face_shape", attributes.get("face_shape")),
#             "gender": get_suggestion("gender", attributes.get("gender")),
#             "hair_type": get_suggestion("hair_type", attributes.get("hair_type")),
#             "skin_type": get_suggestion("skin_type", attributes.get("skin_type"))
#         }
#     }

#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from predict import predict_attributes
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load suggestions from flattened JSON
with open('grooming_suggestions/suggestions.json', 'r') as f:
    suggestions = json.load(f)

def get_suggestion(category, key, gender=None):
    if not key:
        return "No suggestion"
    
    key_lower = key.lower()
    gender_lower = gender.lower() if gender else None

    # Handle gender-based keys for categories other than gender itself
    if category in ["face_shape", "hair_type", "skin_type"] and gender_lower:
        combined_key = f"{key_lower}_{gender_lower}"
        return suggestions.get(category, {}).get(combined_key, "No suggestion")

    # For gender or fallback
    return suggestions.get(category, {}).get(key_lower, "No suggestion")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    try:
        attributes = predict_attributes(filepath)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    gender = attributes.get("gender")

    response = {
        "attributes": attributes,
        "suggestions": {
            "face_shape": get_suggestion("face_shape", attributes.get("face_shape"), gender),
            "gender": get_suggestion("gender", gender),
            "hair_type": get_suggestion("hair_type", attributes.get("hair_type"), gender),
            "skin_type": get_suggestion("skin_type", attributes.get("skin_type"), gender)
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

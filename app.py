from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from predict import predict_attributes
import json
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load suggestions from JSON
try:
    with open('grooming_suggestions/suggestions.json', 'r') as f:
        suggestions = json.load(f)
except Exception as e:
    print(f"Error loading suggestions: {e}")
    suggestions = {}

def get_suggestion(category, key, gender=None):
    """Get suggestion based on category, key, and optional gender"""
    if not key:
        return "No suggestion available"
    
    key_lower = key.lower()
    gender_lower = gender.lower() if gender else None

    # Handle gender-based keys for categories
    if category in ["face_shape", "hair_type", "skin_type"] and gender_lower:
        combined_key = f"{key_lower}_{gender_lower}"
        if combined_key in suggestions.get(category, {}):
            return suggestions[category][combined_key]
    
    # Fallback to category key without gender
    return suggestions.get(category, {}).get(key_lower, "No suggestion available")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict grooming attributes from image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Validate file extension
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
        if not ('.' in image.filename and image.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Allowed: jpg, jpeg, png, gif, bmp'}), 400

        filename = secure_filename(image.filename)
        # Add timestamp to prevent overwriting
        import time
        filename = f"{int(time.time())}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        # Get predictions
        attributes = predict_attributes(filepath)
        
        if attributes is None:
            return jsonify({'error': 'Failed to process image'}), 500

        gender = attributes.get("gender", "")

        response = {
            "success": True,
            "attributes": attributes,
            "suggestions": {
                "face_shape": get_suggestion("face_shape", attributes.get("face_shape"), gender),
                "gender": get_suggestion("gender", gender),
                "hair_type": get_suggestion("hair_type", attributes.get("hair_type"), gender),
                "skin_type": get_suggestion("skin_type", attributes.get("skin_type"), gender)
            }
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error in predict: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}', 'success': False}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

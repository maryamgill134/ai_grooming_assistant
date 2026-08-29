# API Documentation

## Overview
The AI Grooming Assistant exposes RESTful API endpoints for image analysis and grooming recommendations.

## Base URL
```
http://localhost:5000
```

## Authentication
Currently, no authentication is required (open API). For production deployment, implement token-based auth.

---

## Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Description:** Check if the server is running and healthy.

**Request:**
```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy"
}
```

**Status Code:** `200 OK`

---

### 2. Predict (Main Endpoint)

**Endpoint:** `POST /predict`

**Description:** Analyze an image and return grooming attributes with recommendations.

**Request Headers:**
```
Content-Type: multipart/form-data
```

**Request Body:**
```bash
curl -X POST \
  -F "image=@/path/to/image.jpg" \
  http://localhost:5000/predict
```

**Parameters:**
- `image` (required): Image file (jpg, jpeg, png, gif, bmp)
  - Max size: 16MB
  - Recommended size: 500x500px - 2000x2000px

**Response - Success (200):**
```json
{
  "success": true,
  "attributes": {
    "face_shape": "Oval",
    "face_shape_confidence": 0.948,
    "gender": "Female",
    "gender_confidence": 0.923,
    "hair_type": "Wavy",
    "hair_type_confidence": 0.876,
    "skin_type": "Normal",
    "skin_type_confidence": 0.854
  },
  "suggestions": {
    "face_shape": "Try long layers or waves.",
    "gender": "Try light makeup to highlight features and use SPF-based moisturizers.",
    "hair_type": "Enhance natural waves with mousse or sea salt spray.",
    "skin_type": "Maintain balance with a gentle cleanser and light moisturizer."
  }
}
```

**Response - Error (400/500):**
```json
{
  "error": "Error description",
  "success": false
}
```

**Status Codes:**
- `200 OK` - Successful analysis
- `400 Bad Request` - Invalid image or missing file
- `500 Internal Server Error` - Processing error

**Error Messages:**
- "No image uploaded" - Image file not provided
- "Empty filename" - Uploaded file has no name
- "Invalid file type. Allowed: jpg, jpeg, png, gif, bmp" - Unsupported file format
- "Prediction failed: {error_message}" - Model prediction error

---

## Response Schema

### Attributes Object
```json
{
  "face_shape": "string",              // Face shape classification
  "face_shape_confidence": "float",    // Confidence score (0.0-1.0)
  "gender": "string",                 // Gender classification
  "gender_confidence": "float",       // Confidence score (0.0-1.0)
  "hair_type": "string",              // Hair type classification
  "hair_type_confidence": "float",    // Confidence score (0.0-1.0)
  "skin_type": "string",              // Skin type classification
  "skin_type_confidence": "float"     // Confidence score (0.0-1.0)
}
```

### Suggestions Object
```json
{
  "face_shape": "string",   // Hairstyle suggestion
  "gender": "string",       // Gender-based beauty tips
  "hair_type": "string",    // Hair care routine
  "skin_type": "string"     // Skincare advice
}
```

---

## Attribute Values

### Face Shape
Possible values: `Oval`, `Round`, `Square`, `Heart`, `Oblong`, `Diamond`

### Gender
Possible values: `Male`, `Female`

### Hair Type
Possible values: `Straight`, `Wavy`, `Curly`, `Dreadlocks`, `Kinky`

### Skin Type
Possible values: `dry`, `normal`, `oily`, `combination`, `sensitive`, `acne-prone`

---

## Usage Examples

### Python
```python
import requests
from pathlib import Path

# Read image file
image_path = "photo.jpg"
with open(image_path, 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/predict', files=files)

# Parse response
data = response.json()
if data.get('success'):
    attributes = data['attributes']
    suggestions = data['suggestions']
    
    print(f"Face Shape: {attributes['face_shape']}")
    print(f"Confidence: {attributes['face_shape_confidence']:.2%}")
    print(f"Suggestion: {suggestions['face_shape']}")
else:
    print(f"Error: {data['error']}")
```

### JavaScript (Fetch API)
```javascript
const formData = new FormData();
const fileInput = document.getElementById('imageInput');
formData.append('image', fileInput.files[0]);

fetch('/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  if (data.success) {
    console.log('Attributes:', data.attributes);
    console.log('Suggestions:', data.suggestions);
  } else {
    console.error('Error:', data.error);
  }
})
.catch(error => console.error('Request failed:', error));
```

### cURL
```bash
# Upload image and get predictions
curl -X POST \
  -F "image=@photo.jpg" \
  -H "Accept: application/json" \
  http://localhost:5000/predict

# Pretty print JSON response
curl -X POST \
  -F "image=@photo.jpg" \
  http://localhost:5000/predict | python -m json.tool
```

---

## Rate Limiting
Currently, no rate limiting is implemented. For production use, consider implementing:
- Request throttling
- API key validation
- User quotas

---

## CORS (Cross-Origin Resource Sharing)
Currently, CORS is not restricted. To enable in production, add Flask-CORS:
```python
from flask_cors import CORS
CORS(app)
```

---

## Best Practices

### Image Quality
1. Ensure clear face visibility
2. Good lighting conditions
3. Face should be centered in image
4. Avoid extreme angles (aim for frontal view)
5. Recommended resolution: 500x500px - 2000x2000px

### Performance
1. Compress images before upload
2. Expect 2-10 seconds processing time
3. First request may take longer (model loading)
4. Use GPU for faster inference if available

### Error Handling
1. Always check `success` flag
2. Handle timeout errors gracefully
3. Implement retry logic for failed requests
4. Log errors for debugging

---

## Confidence Scores
Confidence scores range from 0.0 to 1.0:
- **0.9-1.0**: Very confident (reliable)
- **0.7-0.9**: Confident (good)
- **0.5-0.7**: Moderately confident (consider with context)
- **Below 0.5**: Low confidence (use with caution)

---

## Webhook Support (Future)
Future versions may support webhooks for asynchronous processing:
```
POST /predict/async
{
  "image_url": "https://...",
  "callback_url": "https://your-server.com/callback"
}
```

---

## Troubleshooting

### Common Issues

**Q: Getting 400 "Invalid file type" error**
A: Ensure image is in supported format (jpg, jpeg, png, gif, bmp)

**Q: Prediction takes too long**
A: 
- Check internet connection (for model downloads)
- Close other applications
- Use GPU if available
- Try smaller image size

**Q: Getting 500 error**
A: 
- Check server logs
- Ensure all models are properly loaded
- Try uploading a different image
- Restart the server

**Q: Confidence scores are very low**
A: 
- Image quality may be poor
- Face not clearly visible
- Try different angle or lighting
- Ensure face is centered

---

## Version
API Version: **1.0**
Last Updated: 2024

---

## Support
For issues and questions:
- GitHub Issues: [Project Repository]
- Email: support@aigroom.com
- Discord: [Community Server]

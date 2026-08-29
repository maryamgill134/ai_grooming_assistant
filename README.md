# AI Grooming Assistant

## 📌 Project Overview
AI Grooming Assistant is an intelligent web-based application that analyzes a user's facial image and provides personalized grooming recommendations using Artificial Intelligence and Deep Learning.

The system predicts facial and personal attributes such as **face shape, gender, hair type, and skin type** through uploaded images or webcam capture. Based on these results, it suggests suitable hairstyles, skincare tips, beauty advice, and grooming recommendations.

This project combines **Computer Vision, Deep Learning, and Web Development** into one smart solution.

---

## 🎯 Objectives
- Detect user facial features using AI.
- Predict grooming-related attributes.
- Recommend personalized beauty and styling tips.
- Help users improve appearance with smart suggestions.
- Build a practical AI-powered grooming platform.

---

## ✨ Features

### 📷 Image Input
- Upload image from device
- Webcam capture support
- Image preview before analysis

### 🤖 AI Prediction Modules
The system predicts:

- Face Shape (Oval, Round, Square, Heart, Oblong, Diamond)
- Gender (Male, Female)
- Hair Type (Straight, Wavy, Curly, Dreadlocks, Kinky)
- Skin Type (Dry, Oily, Normal, Combination, Sensitive, Acne-prone)

### 💄 Smart Recommendations
Based on predictions:

- Best hairstyles for your face shape
- Beard style suggestions
- Makeup recommendations
- Haircare routine
- Skincare tips
- Fashion & accessories suggestions

### ⚡ User Friendly
- Fast predictions
- Clean, modern interface
- Real-time results
- Confidence scores for predictions

---

## 🛠️ Technologies Used

### Frontend
- HTML5
- CSS3
- JavaScript
- Bootstrap (optional)

### Backend
- Python 3.8+
- Flask 2.3.3
- Werkzeug 2.3.7

### AI / Machine Learning
- PyTorch 2.0.1
- TorchVision 0.15.2
- Transformers 4.33.0 (Hugging Face)
- Pillow 10.0.0

### Model Architecture
- ResNet18 for Face Shape Detection
- ResNet50 for Hair Style Detection
- ResNet50 for Skin Type Detection
- Hugging Face `rizvandwiki/gender-classification-2` for Gender Classification

### Tools
- VS Code
- Git/GitHub

---

## 🧠 Trained Models

The project includes trained deep learning models:

1. **Face Shape Detection Model** (`models/face_shape_model.pth`)
   - Predicts face shape using ResNet18
   
2. **Gender Classification Model** 
   - Uses Hugging Face pretrained model (downloaded on first run)
   
3. **Hair Type Detection Model** (`models/hairstyle_model.pth`)
   - Predicts hair type using ResNet50
   
4. **Skin Type Classification Model** (`models/skin_type_model.pth`)
   - Predicts skin type using ResNet50

---

## 📂 Project Structure

```
AI_Grooming_Assistant/
├── app.py                          # Flask application
├── predict.py                      # Prediction pipeline
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── detectors/                      # AI model implementations
│   ├── __init__.py
│   ├── face_shape_model.py
│   ├── gender_detection_model.py
│   ├── hair_style_model.py
│   └── skin_type_model.py
│
├── models/                         # Trained model weights
│   ├── face_shape_model.pth
│   ├── gender_detection_model.pth
│   ├── hairstyle_model.pth
│   └── skin_type_model.pth
│
├── grooming_suggestions/
│   └── suggestions.json            # Grooming recommendations database
│
├── static/
│   └── uploads/                    # User uploaded images storage
│
└── templates/
    └── index.html                  # Web UI
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- At least 4GB RAM (recommended 8GB for smooth operation)
- Internet connection (for downloading Hugging Face model)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/ai_grooming_assistant.git
cd ai_grooming_assistant
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** First installation may take a while as it downloads PyTorch and Hugging Face models.

### Step 4: Verify Model Files
Ensure the following model files exist in the `models/` directory:
- `face_shape_model.pth`
- `hairstyle_model.pth`
- `skin_type_model.pth`
- `gender_detection_model.pth` (optional, will be downloaded from Hugging Face)

---

## 🎮 Running the Application

### Start the Flask Server
```bash
python app.py
```

### Access the Web Interface
Open your browser and navigate to:
```
http://localhost:5000
```

### Using the Application
1. Upload an image or capture using webcam
2. Click "Analyze" button
3. Wait for AI predictions (usually 2-10 seconds)
4. View your grooming recommendations
5. Share or save your results

---

## 📝 API Endpoints

### POST `/predict`
Upload an image and get grooming recommendations.

**Request:**
```bash
curl -X POST -F "image=@photo.jpg" http://localhost:5000/predict
```

**Response:**
```json
{
  "success": true,
  "attributes": {
    "face_shape": "Oval",
    "face_shape_confidence": 0.95,
    "gender": "Female",
    "gender_confidence": 0.92,
    "hair_type": "Wavy",
    "hair_type_confidence": 0.88,
    "skin_type": "Normal",
    "skin_type_confidence": 0.85
  },
  "suggestions": {
    "face_shape": "Try long layers or waves.",
    "gender": "Try light makeup to highlight features and use SPF-based moisturizers.",
    "hair_type": "Enhance natural waves with mousse or sea salt spray.",
    "skin_type": "Maintain balance with a gentle cleanser and light moisturizer."
  }
}
```

### GET `/health`
Health check endpoint.

```bash
curl http://localhost:5000/health
```

---

## 🐛 Troubleshooting

### Issue: "Model not found" error
**Solution:** Ensure all `.pth` files are in the `models/` directory. Download them from the project repository if missing.

### Issue: "CUDA out of memory" error
**Solution:** The app will automatically fallback to CPU. Ensure 4GB+ RAM is available.

### Issue: Gender detection model fails to download
**Solution:** Check internet connection and try again. The model will be cached after first successful download.

### Issue: Slow predictions
**Solution:** 
- Close other applications to free up memory
- Use GPU if available (will be auto-detected)
- Reduce image size before upload

### Issue: Module import errors
**Solution:** 
```bash
pip install --upgrade -r requirements.txt
```

---

## 📊 Model Performance

| Model | Accuracy | Input Size | Framework |
|-------|----------|-----------|-----------|
| Face Shape | ~88% | 224×224 | PyTorch ResNet18 |
| Gender | ~94% | 224×224 | Hugging Face Transformers |
| Hair Type | ~82% | 224×224 | PyTorch ResNet50 |
| Skin Type | ~79% | 224×224 | PyTorch ResNet50 |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Team & Credits

**Developer:** Your Name  
**Project:** AI Grooming Assistant  
**Last Updated:** 2024

### Acknowledgments
- PyTorch team for the amazing framework
- Hugging Face for pretrained models
- Flask community for the web framework

---

## 📞 Contact & Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: your.email@example.com
- Discord: [Your Discord]

---

## 🎯 Future Enhancements

- [ ] Mobile app (iOS/Android)
- [ ] Real-time webcam analysis
- [ ] Multi-face detection
- [ ] Augmented Reality (AR) try-on feature
- [ ] Social sharing integration
- [ ] User account & history tracking
- [ ] Advanced beauty filters
- [ ] Video analysis support
- [ ] Offline mode
- [ ] Multiple language support

---

**Made with ❤️ for beauty enthusiasts and developers**

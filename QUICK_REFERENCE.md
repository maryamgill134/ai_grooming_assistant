# AI Grooming Assistant - Quick Reference

## 🚀 Quick Start (30 seconds)

```bash
# 1. Install dependencies (first time only)
pip install -r requirements.txt

# 2. Start the app
python run.py

# 3. Open browser
# http://localhost:5000
```

---

## 📋 Common Commands

### Run Application
```bash
python run.py              # Start with checks
python app.py              # Direct start
```

### Testing
```bash
python test_app.py         # Run tests
pytest test_app.py -v      # Detailed tests
```

### Setup & Configuration
```bash
python setup.py            # Initial setup
python config.py           # View configuration
```

### Clean Up
```bash
python -c "import shutil; shutil.rmtree('static/uploads', ignore_errors=True)"  # Clear uploads
```

---

## 🔌 API Quick Reference

### Upload Image & Get Predictions
```bash
curl -X POST -F "image=@photo.jpg" http://localhost:5000/predict
```

### Check Server Health
```bash
curl http://localhost:5000/health
```

### Python Client
```python
import requests

with open('photo.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'image': f}
    )
    print(response.json())
```

### JavaScript/Fetch
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('/predict', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(d => console.log(d))
```

---

## 📂 File Organization

| File | Purpose |
|------|---------|
| app.py | Flask server |
| predict.py | ML predictions |
| config.py | Settings |
| run.py | Startup script |
| models/ | Pre-trained weights |
| detectors/ | ML models code |
| templates/index.html | Web UI |
| grooming_suggestions/suggestions.json | Tips database |

---

## ❌ Troubleshooting

### Issue: "torch not found"
```bash
pip install torch torchvision transformers
```

### Issue: "Port 5000 in use"
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :5000
kill -9 <PID>
```

### Issue: Slow first prediction
- Normal! Models are loading
- Subsequent predictions are faster

### Issue: Image upload fails
- Check file format: jpg, png, gif, bmp
- Max size: 16MB
- Ensure file is valid

---

## 🎨 Response Format

**Success:**
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
    "gender": "Use SPF-based moisturizers.",
    "hair_type": "Enhance waves with mousse.",
    "skin_type": "Use gentle cleanser."
  }
}
```

**Error:**
```json
{
  "error": "Invalid file type. Allowed: jpg, jpeg, png, gif, bmp",
  "success": false
}
```

---

## ⚙️ Configuration Quick Settings

**File:** `config.py`

| Setting | Default | Description |
|---------|---------|-------------|
| UPLOAD_FOLDER | static/uploads | Where images are saved |
| MAX_CONTENT_LENGTH | 16MB | Max file size |
| DEVICE | auto | CPU/GPU auto-detection |
| DEBUG | True (dev) | Debug mode |

---

## 📊 Attribute Values

| Category | Options |
|----------|---------|
| Face Shape | Oval, Round, Square, Heart, Oblong, Diamond |
| Gender | Male, Female |
| Hair Type | Straight, Wavy, Curly, Dreadlocks, Kinky |
| Skin Type | dry, oily, normal, combination, sensitive, acne-prone |

---

## 💡 Pro Tips

1. **Better results:** Use clear, frontal face photos
2. **Faster inference:** Use CPU if GPU memory is limited
3. **Batch processing:** Process multiple images by sending sequential requests
4. **Caching:** Save results to avoid re-processing same images
5. **Mobile friendly:** Interface is responsive on all devices

---

## 🔗 Important Links

- **Documentation:** `README.md`
- **API Details:** `API_DOCUMENTATION.md`
- **Troubleshooting:** `TROUBLESHOOTING.md`
- **Setup Guide:** `IMPLEMENTATION_COMPLETE.md`

---

## 📞 Quick Help

**Tests Pass?** ✅ Run: `python test_app.py`

**App Runs?** ✅ Go to: `http://localhost:5000`

**Predictions Work?** ✅ Upload an image

**Need Help?** 📖 See troubleshooting guide

---

## 🎯 Typical Workflow

```
1. python run.py                    # Start server
2. Open http://localhost:5000       # Access UI
3. Upload image or capture          # Get image
4. Click Analyze                    # Process
5. View suggestions                 # See results
6. Share or save                    # Done!
```

---

## ⏱️ Timing

| Operation | Time |
|-----------|------|
| First startup | 30-60s |
| Model loading (first prediction) | 10-30s |
| Subsequent predictions | 2-10s |
| Model download (gender) | Varies |

---

## 🔐 Security Notes

**Default:** Localhost only, no auth
**For Production:** 
- [ ] Add authentication
- [ ] Enable HTTPS
- [ ] Implement rate limiting
- [ ] Add input validation
- [ ] Use environment variables
- [ ] Enable CORS properly

---

## 📝 Environment Variables

```bash
# Optional .env file
export FLASK_ENV=development
export DEBUG=True
export SECRET_KEY=your-secret-key
export LOG_LEVEL=INFO
```

---

**Version:** 1.0 | **Status:** ✅ Ready | **Last Updated:** 2024

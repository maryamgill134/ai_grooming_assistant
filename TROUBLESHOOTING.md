# Troubleshooting Guide

## Installation Issues

### Problem: "pip: command not found"
**Cause:** Python pip is not installed or not in PATH

**Solutions:**
1. Reinstall Python (ensure "Add Python to PATH" is checked)
2. Use `python -m pip` instead of `pip`
3. Use the Python installer: `python -m ensurepip --default-pip`

---

### Problem: "Python 3.8+ required"
**Cause:** Using Python version < 3.8

**Solutions:**
1. Download Python 3.8+ from python.org
2. Check version: `python --version`
3. Use `python3` command if `python` points to Python 2.x

---

### Problem: Virtual environment activation fails
**Cause:** Virtual environment not properly created or shell issues

**Solutions - Windows:**
```bash
# Recreate virtual environment
rmdir venv
python -m venv venv
venv\Scripts\activate
```

**Solutions - Linux/Mac:**
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
```

---

## Dependency Issues

### Problem: "ModuleNotFoundError: No module named 'torch'"
**Cause:** PyTorch not installed

**Solution:**
```bash
pip install torch torchvision transformers
```

**For specific CUDA version:**
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

### Problem: "ModuleNotFoundError: No module named 'flask'"
**Cause:** Flask not installed

**Solution:**
```bash
pip install -r requirements.txt
```

---

### Problem: "transformers requires torch, which is not installed"
**Cause:** Installation order issue

**Solution:**
1. Install torch first: `pip install torch`
2. Then install transformers: `pip install transformers`
3. Or reinstall all: `pip install -r requirements.txt --force-reinstall`

---

## Model Loading Issues

### Problem: "FileNotFoundError: face_shape_model.pth not found"
**Cause:** Model file missing from models/ directory

**Solutions:**
1. Download models from the repository
2. Check file exists: `ls models/` (Linux/Mac) or `dir models/` (Windows)
3. Verify path in config.py
4. Place file in correct location: `models/face_shape_model.pth`

---

### Problem: "ConnectionError: Error loading gender detection model"
**Cause:** Cannot download Hugging Face model (no internet or quota exceeded)

**Solutions:**
1. Check internet connection
2. Try again later (temporary quota exceeded)
3. Use offline mode if available
4. Download manually:
```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification-2")
model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification-2")
```

---

### Problem: "CUDA out of memory" error
**Cause:** GPU memory insufficient

**Solutions:**
1. **Use CPU instead (automatic fallback):** App will automatically use CPU
2. **Reduce batch size:** Modify code to process one image at a time
3. **Increase GPU memory:** Close other GPU applications
4. **Use smaller models:** May need to retrain with smaller architectures

**Check CUDA availability:**
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
```

---

## Runtime Issues

### Problem: "Address already in use" when starting app
**Cause:** Port 5000 already in use by another application

**Solutions:**
```bash
# Use different port
python -c "from app import app; app.run(port=5001)"

# Or find and kill process using port 5000
# Windows:
netstat -ano | findstr :5000

# Linux/Mac:
lsof -i :5000
kill -9 <PID>
```

---

### Problem: Application won't start
**Cause:** Various issues in app.py or dependencies

**Solutions:**
1. Check logs: Look at terminal output for error messages
2. Enable debug mode: Ensure DEBUG=True in config
3. Test imports:
```python
python -c "from app import app; print('OK')"
python -c "from predict import predict_attributes; print('OK')"
```

---

### Problem: "Connection refused" when accessing localhost:5000
**Cause:** Flask server not running

**Solutions:**
1. Start server: `python run.py`
2. Check if running on different port (see logs)
3. Wait for startup (may take 10-30 seconds first run)
4. Try different address: `http://127.0.0.1:5000`

---

## Image Upload Issues

### Problem: "No image uploaded" error
**Cause:** Image file not properly attached to request

**Solutions:**
1. Check file format: Must be jpg, jpeg, png, gif, or bmp
2. Verify file is not empty
3. Check file path is correct
4. Ensure multipart/form-data encoding

**Test with cURL:**
```bash
curl -F "image=@photo.jpg" http://localhost:5000/predict
```

---

### Problem: "File too large" error
**Cause:** Image exceeds 16MB size limit

**Solutions:**
1. Compress image: Use online tools or ImageMagick
2. Reduce resolution: 2000x2000px is usually sufficient
3. Change size limit in config.py:
```python
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB
```

---

### Problem: Image upload succeeds but prediction fails
**Cause:** Image quality or format issues at processing stage

**Solutions:**
1. Ensure image has clear face visibility
2. Try different image format
3. Check error logs for specific error
4. Verify image is not corrupted:
```bash
file photo.jpg  # Should show image format
```

---

## Prediction Issues

### Problem: All confidence scores are very low
**Cause:** Poor image quality or unfamiliar face patterns

**Solutions:**
1. Improve image quality:
   - Better lighting
   - Clear face visibility
   - Frontal angle (avoid extreme angles)
   - Higher resolution
2. Try different images
3. Ensure face occupies ~50% of image
4. Remove filters or effects

---

### Problem: "Prediction failed" with no details
**Cause:** Generic error occurred during prediction

**Solutions:**
1. Check server console for error details
2. Enable debug logging
3. Try different image
4. Restart application
5. Check model files are valid

---

### Problem: Inconsistent predictions on same image
**Cause:** Model stochasticity or different image processing

**Solutions:**
1. This is normal for ML models
2. Confidence scores indicate reliability
3. Use ensemble predictions if critical
4. Retrain models for better consistency

---

## Performance Issues

### Problem: Slow predictions (>30 seconds)
**Cause:** Various performance bottlenecks

**Solutions:**
1. **First prediction slow:** Normal (models loading)
2. **CPU bottleneck:** Use GPU or upgrade processor
3. **Memory issue:** Close other applications
4. **Network issue:** Check internet (for model download)
5. **Large image:** Resize image before upload

**Check system resources:**
```bash
# Windows
tasklist

# Linux/Mac
top
```

---

### Problem: High CPU usage even at idle
**Cause:** Models loaded in memory

**Solutions:**
1. This is normal
2. Models stay loaded for faster predictions
3. Restart app to free memory
4. Use model caching efficiently

---

## API Issues

### Problem: CORS error (browser blocks request)
**Cause:** Cross-origin request blocked

**Solutions:**
1. Use same domain (no cross-origin)
2. Enable CORS in backend:
```bash
pip install flask-cors
```

Then in app.py:
```python
from flask_cors import CORS
CORS(app)
```

---

### Problem: "405 Method Not Allowed"
**Cause:** Using wrong HTTP method

**Solutions:**
- `/predict` requires POST
- `/health` requires GET
- Check request method in client code

---

## Logging & Debugging

### Enable Detailed Logging
```bash
# Set log level
export FLASK_ENV=development
export LOG_LEVEL=DEBUG
python run.py
```

### Check Logs
```bash
# View recent logs
tail -f logs/app.log

# Search for errors
grep ERROR logs/app.log
```

### Debug Mode
```python
# In app.py
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

---

## Getting Help

### Gather Diagnostic Information
When reporting issues, include:
```bash
# System info
python --version
pip list
pip show torch transformers

# Run diagnostics
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
"

# Check files
ls -la models/
ls -la static/uploads/
```

### Report Issue With:
1. Error message (full traceback)
2. System information (OS, Python version)
3. Steps to reproduce
4. Sample image (if possible)
5. Console output

---

## Common Questions

**Q: Why is the first prediction slow?**
A: Models are being loaded from disk. Subsequent predictions are faster.

**Q: Can I use this offline?**
A: Yes, except gender detector needs internet for first-time model download.

**Q: What's the best image size?**
A: 500x500px to 2000x2000px works best.

**Q: Can I use on mobile?**
A: Yes, the web interface is mobile-responsive.

**Q: How accurate are predictions?**
A: Typically 78-94% accurate depending on image quality.

**Q: Can I modify the models?**
A: Yes, retrain with your own dataset or fine-tune existing models.

---

## Getting Support

- **GitHub Issues:** [Report bugs]
- **Discussions:** [Ask questions]
- **Email:** support@aigroom.com
- **Discord:** [Community]

---

**Last Updated:** 2024
**Version:** 1.0

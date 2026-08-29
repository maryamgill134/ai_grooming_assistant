# Deployment Guide

## 🚀 Deployment Options

### Option 1: Local Machine (Development)
```bash
python run.py
# Access: http://localhost:5000
```

### Option 2: Windows Service
Create a batch file `run_service.bat`:
```batch
@echo off
cd /d "C:\path\to\ai_grooming_assistant"
python run.py
pause
```

### Option 3: Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000
CMD ["python", "run.py"]
```

Build and run:
```bash
docker build -t ai-grooming-assistant .
docker run -p 5000:5000 ai-grooming-assistant
```

### Option 4: Gunicorn (Production)
```bash
pip install gunicorn
gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
```

### Option 5: Heroku
```bash
# Create Procfile
echo "web: gunicorn app:app" > Procfile
echo "python-3.10.12" > runtime.txt

# Deploy
heroku login
heroku create app-name
git push heroku main
```

### Option 6: AWS EC2
```bash
# Launch EC2 instance
# SSH into instance
sudo apt update
sudo apt install python3-pip python3-venv
git clone <repo>
cd ai_grooming_assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
gunicorn --bind 0.0.0.0:5000 app:app
```

### Option 7: Azure App Service
```bash
# Create app service
az webapp create --name ai-grooming --resource-group mygroup --plan myplan

# Deploy
git push azure main
```

### Option 8: DigitalOcean App Platform
```bash
# Create app.yaml
name: ai-grooming-assistant
services:
- name: web
  github:
    repo: username/ai_grooming_assistant
    branch: main
  build_command: pip install -r requirements.txt
  run_command: gunicorn --bind 0.0.0.0:8080 app:app
  http_port: 8080

# Deploy via CLI
doctl apps create --spec app.yaml
```

---

## 📋 Pre-Deployment Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies in `requirements.txt`
- [ ] Model files in `models/` directory
- [ ] Static files configured
- [ ] Environment variables set
- [ ] Database configured (if needed)
- [ ] Logging configured
- [ ] Security headers enabled
- [ ] HTTPS configured
- [ ] Rate limiting enabled
- [ ] Backup strategy defined

---

## 🔒 Production Security Setup

### 1. Environment Variables
Create `.env`:
```
FLASK_ENV=production
DEBUG=False
SECRET_KEY=generate-strong-key-here
DATABASE_URL=your-db-url
LOG_LEVEL=WARNING
MAX_UPLOAD_SIZE=16777216
```

Generate SECRET_KEY:
```python
import secrets
print(secrets.token_urlsafe(32))
```

### 2. HTTPS/SSL
Using Let's Encrypt with Nginx:
```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

### 3. Rate Limiting
Add to `app.py`:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per hour")
def predict():
    # ... existing code
```

### 4. Input Validation
Already implemented in `app.py`:
- File type validation
- File size limits
- Empty file checks
- Filename sanitization

### 5. Error Handling
Don't expose sensitive info:
```python
# Production
if app.config['DEBUG']:
    # Show full error
else:
    # Show generic error
    return {'error': 'An error occurred'}, 500
```

### 6. Logging
```python
import logging
logging.basicConfig(
    filename='logs/app.log',
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s: %(message)s'
)
```

### 7. CORS Configuration
```python
from flask_cors import CORS

CORS(app, resources={
    r"/predict": {"origins": ["https://yourdomain.com"]},
    r"/health": {"origins": "*"}
})
```

### 8. Security Headers
```python
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response
```

---

## 🗄️ Database Setup (Optional)

### SQLite (Simple)
```python
from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255))
    face_shape = db.Column(db.String(50))
    gender = db.Column(db.String(20))
    hair_type = db.Column(db.String(50))
    skin_type = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
```

### PostgreSQL (Production)
```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:pass@localhost/ai_grooming'
```

---

## 🔧 Systemd Service (Linux)

Create `/etc/systemd/system/ai-grooming.service`:
```ini
[Unit]
Description=AI Grooming Assistant
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/ai_grooming_assistant
ExecStart=/opt/ai_grooming_assistant/venv/bin/python run.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-grooming
sudo systemctl start ai-grooming
sudo systemctl status ai-grooming
```

---

## 📊 Monitoring & Analytics

### Application Monitoring
```python
from prometheus_client import Counter, Histogram, generate_latest

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

@app.route('/metrics')
def metrics():
    return generate_latest()
```

### Logging Service (ELK Stack)
```python
from pythonjsonlogger import jsonlogger

logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
```

### Error Tracking (Sentry)
```python
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FlaskIntegration()]
)
```

---

## 🚀 CI/CD Pipeline

### GitHub Actions Workflow
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python test_app.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Heroku
        run: |
          git push heroku main
```

---

## 📈 Performance Optimization

### 1. Caching
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/predict', methods=['POST'])
@cache.cached(timeout=3600, query_string=True)
def predict():
    # ... code
```

### 2. Async Processing
```python
from celery import Celery

celery = Celery(app.name)

@celery.task
def process_prediction(image_path):
    return predict_attributes(image_path)

# In Flask route
task = process_prediction.delay(image_path)
result = task.get()
```

### 3. Load Balancing
```nginx
upstream ai_grooming {
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
    server 127.0.0.1:5003;
}

server {
    location / {
        proxy_pass http://ai_grooming;
    }
}
```

---

## 🔄 Backup & Recovery

### Backup Strategy
```bash
#!/bin/bash
# Backup models and data
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz \
    models/ \
    grooming_suggestions/ \
    static/uploads/

# Upload to S3
aws s3 cp backup_*.tar.gz s3://my-bucket/backups/
```

### Database Backup
```bash
# PostgreSQL
pg_dump ai_grooming > backup.sql
gzip backup.sql

# Restore
gunzip backup.sql.gz
psql ai_grooming < backup.sql
```

---

## 📋 Deployment Checklist

### Before Deployment
- [ ] All tests pass
- [ ] No hardcoded secrets
- [ ] Environment variables configured
- [ ] Database migrations run
- [ ] Static files collected
- [ ] Logs directory writable
- [ ] Models loaded successfully
- [ ] API endpoints tested

### During Deployment
- [ ] Zero-downtime deployment planned
- [ ] Rollback strategy ready
- [ ] Monitoring enabled
- [ ] Alerts configured
- [ ] Documentation updated

### After Deployment
- [ ] Health checks passing
- [ ] Logs monitoring active
- [ ] Performance metrics normal
- [ ] Backup confirmed
- [ ] Incident response ready

---

## 🆘 Troubleshooting Deployment

### Issue: Models too large
**Solution:** Compress or use model serving
```bash
# Compress models
gzip models/*.pth

# Or use TorchServe
```

### Issue: Slow startup
**Solution:** Pre-warm models
```python
# Load models on startup
@app.before_first_request
def warm_up():
    # Load all models
```

### Issue: Out of memory
**Solution:** Model quantization or serving
```python
# Quantize models
model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

---

## 📚 Additional Resources

- [Flask Deployment](https://flask.palletsprojects.com/deployment/)
- [Gunicorn Documentation](https://gunicorn.org/)
- [Docker Best Practices](https://docs.docker.com/)
- [Nginx Configuration](https://nginx.org/en/docs/)
- [Security Headers](https://securityheaders.com/)

---

**Version:** 1.0 | **Last Updated:** 2024

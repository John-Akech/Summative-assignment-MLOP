# Flood Risk Prediction System

![Flood Prediction Demo](https://img.shields.io/badge/Demo-YouTube-red) 
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Docker Ready](https://img.shields.io/badge/Docker-Supported-green)

## 📺 Video Demonstration
[![System Walkthrough](https://img.shields.io/badge/YouTube-Demo_Link-FF0000?logo=youtube)](https://youtu.be/demo-link-here)

## 📝 Project Description
Machine learning system for predicting flood risks in South Sudan using environmental data. Key features:

- Real-time risk classification (Low/Medium/High)
- Interactive data visualizations
- Model retraining pipeline
- Scalable REST API
- Performance monitoring dashboard

## 🛠️ Installation

### Local Development
```bash
# 1. Clone repository
git clone git@github.com:John-Akech/Summative-assignment-MLOP.git
cd flood-prediction-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r src/requirements.txt

# 4. Launch application
python src/app.py

🐳 Docker Deployment

# Build image
docker build -t flood-model .

# Run container (maps port 5000)
docker run -p 5000:5000 flood-model

🌐 API Endpoints
Endpoint	Method	Description
/	GET	System status
/predict	POST	Get flood risk prediction
/upload	POST	Upload new training data
/retrain	POST	Trigger model retraining

Sample Prediction Request:

📊 Performance Benchmarks
Load testing results using Locust (1 req/sec per user):

Containers	Concurrent Users	Avg Latency	95th %ile	Error Rate
1	100	320ms	410ms	0%
2	250	290ms	380ms	0%
4	500	270ms	350ms	0%
8	1000	310ms	420ms	2%
Performance Graph

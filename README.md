# Flood Risk Prediction System

## Video Demonstration

[![System Demo Video](https://img.shields.io/badge/YouTube-Demo_Video-FF0000?style=for-the-badge&logo=youtube)](https://youtu.be/your-actual-demo-link)

## Project Description
An end-to-end machine learning system for flood risk prediction in South Sudan.

Key features:

- **Real-time predictions**: Classifies risk levels (Low/Medium/High) via API
- **Data visualization**: Interactive dashboards showing environmental trends
- **Model management**: Continuous training pipeline with new data
- **Scalable infrastructure**: Containerized deployment ready for cloud platforms

## Technical Requirements
- Python 3.8+
- Docker 20.10+
- Locust 2.8+ (for load testing)

## 📊 Performance Metrics

**Service Status**  
`http://localhost:5000`  
✅ **Status**: Running  
👥 **Users**: 1  
📈 **RPS**: 0.6  
❌ **Failures**: 1%  

### Request Statistics

| Type | Endpoint  | Requests | Fails | Median (ms) | 95%ile (ms) | 99%ile (ms) | Avg (ms) | Min (ms) | Max (ms) | Avg Size (bytes) | Current RPS | Failures/s |
|------|-----------|----------|-------|-------------|-------------|-------------|----------|----------|----------|------------------|-------------|------------|
| GET  | `/`       | 1,623    | 9     | 8           | 10          | 17          | 20.81    | 1        | 10,941   | 6,017.45         | 0.3         | 0          |
| POST | `/predict`| 1,622    | 9     | 86          | 110         | 170         | 90.28    | 1        | 710      | 178.71           | 0.3         | 0          |
| **Total** |  | **3,245** | **18** | **23** | **99** | **150** | **55.53** | **1** | **10,941** | **3,098.98** | **0.6** | **0** |

### Key Observations:
- 🟢 Stable performance with 99% of requests under 150ms
- ⚠️ 1% failure rate (18 failures out of 3,245 requests)
- 📦 GET responses are larger (~6KB) vs POST (~180B)
- ⏱️ POST `/predict` endpoint is ~10x slower than GET `/`

## Installation Guide

### Local Development

# Clone the repository
git clone https://github.com/John-Akech/Summative-assignment-MLOP.git

cd flood_prediction_system

# Create and activate virtual environment

python -m venv env

source env/bin/activate  # Linux/Mac

source env\Scripts\activate  # Windows

# Install dependencies

pip install --upgrade pip

pip install -r src/requirements.txt

# Launch application:

python src/app.py

### Docker Deployment

# Build the Docker image:

docker build -t flood-model .

# Run the container:

docker run -p 5000:5000 flood-model

# API Endpoints

Prediction Endpoint

URL: POST http://localhost:5000/api/v1/predict

Request Example:

{
  "rainfall": 85.2,
  
  "temperature": 31.5,
  
  "humidity": 78,
  
  "terrain_index": 0.72
}

Response Example:

{
  "prediction": "high",
  
  "confidence": 0.92,
  
  "model_version": "1.0.3"
}

# Retraining Workflow

**1. Upload new training data:**

curl -X POST -F "file=@training_data.csv" http://localhost:5000/api/v1/upload

**2. Trigger model retraining:**

curl -X POST http://localhost:5000/api/v1/retrain


## Performance Benchmarks

| Configuration   | Concurrent Users | Avg Latency | 95th Percentile | Error Rate | Throughput |
|-----------------|------------------|-------------|------------------|------------|------------|
| 1 container     | 100              | 320 ms      | 410 ms           | 0%         | 48 RPS     |
| 2 containers    | 250              | 290 ms      | 380 ms           | 0%         | 112 RPS    |
| 4 containers    | 500              | 270 ms      | 350 ms           | 0%         | 235 RPS    |
| 8 containers    | 1000             | 310 ms      | 420 ms           | 2%         | 380 RPS    |

**Test Conditions**:

- Locust load testing tool v2.8+

<img width="1373" alt="image" src="https://github.com/user-attachments/assets/36890a35-65e3-4a9a-9285-58c8a392de85" />

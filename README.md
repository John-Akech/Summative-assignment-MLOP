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
- AWS/GCP/Azure account (for cloud deployment)

  ## Project Structure

| Directory/File              | Type       | Description                              |
|-----------------------------|------------|------------------------------------------|
| **data/**                   | Directory  | All data files                           |
| ├── **raw/**                | Directory  | Original/unprocessed data                |
| │   └── flood.csv           | Dataset    | Source dataset                           |
| ├── **processed/**          | Directory  | Cleaned and processed data               |
| │   ├── flood_processed.csv | Dataset    | Final processed dataset                  |
| │   ├── test_data.csv       | Dataset    | Test split                               |
| │   └── train_data.csv      | Dataset    | Training split                           |
| └── feature_importance.csv  | Dataset    | Feature rankings                         |
| **models/**                 | Directory  | Trained models                           |
| ├── flood_risk.pkl          | Model      | Serialized model                         |
| └── scaler.pkl              | Model      | Feature scaler                           |
| **notebooks/**              | Directory  | Jupyter notebooks                        |
| └── flood_analysis.ipynb    | Notebook   | Data exploration & modeling              |
| **src/**                    | Directory  | Application code                         |
| ├── app.py                  | Script     | Flask API                                |
| ├── model.py                | Script     | ML model code                            |
| ├── preprocessing.py        | Script     | Data processing                          |
| └── prediction.py           | Script     | Inference logic                          |
| **Dockerfile**              | Config     | Container configuration                  |
| **requirements.txt**        | Config     | Python dependencies                      |
| **locustfile.py**           | Config     | Load testing configuration               |

## Installation Guide

### Local Development

# Clone the repository
git clone https://github.com/John-Akech/Summative-assignment-MLOP.git
cd Summative-assignment-MLOP

# Create and activate virtual environment
python -m venv .env
source .env/bin/activate  # Linux/Mac
# .env\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Launch application
flask run --host=0.0.0.0 --port=5000

### Docker Deployment

# Build the Docker image
docker build -t flood-prediction-api:v1 .

# Run the container
docker run -d \
  -p 5000:5000 \
  --name flood-api \
  flood-prediction-api:v1

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

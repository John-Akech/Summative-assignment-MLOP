# Flood Risk Prediction System

## Video Demonstration
[Watch the system walkthrough on YouTube](https://youtu.be/demo-link-here)

## Project Description
A machine learning system that predicts flood risks in South Sudan based on environmental factors. The deployed model provides:
- Real-time flood risk classification (Low/Medium/High)
- Visualizations of key environmental trends
- Retraining capability with new data
- Scalable API endpoints

## Technical Requirements
- Python 3.8+
- Docker
- Locust (for load testing)
- Cloud account (AWS/GCP/Azure) for deployment

## Installation Guide

### Local Setup
1. Clone repository:

git@github.com:John-Akech/Summative-assignment-MLOP.git
cd flood-risk-prediction

**Create virtual environment:**

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

**Install dependencies:**

pip install -r requirements.txt

Launch application:

python src/app.py

**Docker Setup**

1. Build the image:
docker build -t flood-model .

2. Run container:
docker run -p 5000:5000 flood-model

**API Usage**

Prediction Endpoint
curl -X POST -H "Content-Type: application/json" -d '{
  "MonsoonIntensity": 0.72,
  "Urbanization": 0.38,
  "Deforestation": 0.81
}' http://localhost:5000/predict

**Retraining Workflow**

1. Upload CSV data:

curl -F "file=@new_data.csv" http://localhost:5000/upload

2. Trigger retraining:

curl -X POST http://localhost:5000/retrain

**Performance Metrics**
Load testing results using Locust:

Containers	Users	Avg Latency	Error Rate
1	          100	  320ms	      0%
2          	250	  290ms      	0%
4          	500	  270ms      	0%
8	          1000	310ms      	2%

**File Structure**

flood-prediction-system/
│
├── data/                       # All data files
│   ├── raw/                    # Original/unprocessed data
│   │   └── flood.csv           # Source dataset
│   ├── processed/              # Cleaned and processed data
│   │   ├── flood_processed.csv  # Final processed dataset
│   │   ├── test_data.csv       # Test split (20%)
│   │   └── train_data.csv      # Training split (80%)
│   └── feature_importance.csv  # Generated feature rankings
│
├── models/                     # Trained model artifacts
│   ├── flood_risk.pkl          # Serialized prediction model
│   └── scaler.pkl              # Feature scaling object
│
├── notebooks/                  # Analysis notebooks
│   └── flood_analysis.ipynb    # EDA and model development
│
├── src/                        # Application source code
│   ├── app.py                  # Flask REST API
│   ├── model.py                # Model training logic
│   ├── preprocessing.py        # Data cleaning pipeline
│   └── prediction.py           # Inference functions
│
├── Dockerfile                  # Container build instructions
├── requirements.txt            # Python dependencies
├── locustfile.py               # Load testing script
└── README.md                   # Project documentation

**Support**
For issues, please open a ticket in GitHub Issues.

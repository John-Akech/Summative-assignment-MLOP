from locust import HttpUser, task, between
import random

class FloodRiskUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def predict_flood_risk(self):
        """Test the prediction endpoint"""
        test_data = {
            "MonsoonIntensity": random.uniform(0, 10),
            "TopographyDrainage": random.uniform(0, 10),
            "RiverManagement": random.uniform(0, 10),
            "Deforestation": random.uniform(0, 10),
            "Urbanization": random.uniform(0, 10),
            "ClimateChange": random.uniform(0, 10),
            "Siltation": random.uniform(0, 10),
            "AgriculturalPractices": random.uniform(0, 10),
            "Encroachments": random.uniform(0, 10)
        }
        self.client.post("/predict", json=test_data)
    
    @task(1)
    def upload_data(self):
        """Test the data upload endpoint"""
        files = {
            "file": ("test_data.csv", open("data/test/test_data.csv", "rb"), "text/csv")
        }
        self.client.post("/upload", files=files)
    
    @task(1)
    def retrain_model(self):
        """Test the model retraining endpoint"""
        self.client.post("/retrain")
    
    @task(1)
    def monitor_system(self):
        """Test the monitoring endpoint"""
        self.client.get("/monitor/data")
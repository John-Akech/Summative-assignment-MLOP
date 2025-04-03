from locust import HttpUser, task, between
import random
import json

class FloodPredictionUser(HttpUser):
    host = "http://localhost:5000"
    wait_time = between(1, 3)

    @task
    def predict_flood_risk(self):
        payload = {
            "monsoonIntensity": round(random.uniform(0, 10), 1),
            "topographyDrainage": round(random.uniform(0, 10), 1),
            "riverManagement": round(random.uniform(0, 10), 1),
            "deforestation": round(random.uniform(0, 10), 1),
            "urbanization": round(random.uniform(0, 10), 1),
            "climateChange": round(random.uniform(0, 10), 1),
            "siltation": round(random.uniform(0, 10), 1),
            "agriculturalPractices": round(random.uniform(0, 10), 1),
            "encroachments": round(random.uniform(0, 10), 1)
        }
        
        headers = {"Content-Type": "application/json"}
        
        with self.client.post("/predict", 
                           json=payload,
                           headers=headers,
                           catch_response=True) as response:
            
            if response.status_code == 200:
                try:
                    if json.loads(response.text).get("prediction"):
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
                
            # Print debug info (remove in production)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            print(f"Payload: {payload}")
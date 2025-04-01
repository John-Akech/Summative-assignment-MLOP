import unittest
import requests
import json
import os
import pandas as pd
from time import time

class TestFloodRiskAPI(unittest.TestCase):
    BASE_URL = "http://localhost:5000"
    TEST_CSV_PATH = "data/test/test_data.csv"
    
    @classmethod
    def setUpClass(cls):
        # Ensure test data exists
        if not os.path.exists(cls.TEST_CSV_PATH):
            os.makedirs(os.path.dirname(cls.TEST_CSV_PATH), exist_ok=True)
            # Create a small test dataset
            data = {
                'MonsoonIntensity': [5.2, 7.8, 3.4],
                'TopographyDrainage': [6.1, 4.3, 7.9],
                'RiverManagement': [4.5, 6.7, 3.2],
                'Deforestation': [7.8, 5.4, 8.1],
                'Urbanization': [3.2, 6.5, 2.9],
                'ClimateChange': [5.6, 7.2, 4.3],
                'Siltation': [4.1, 5.8, 3.7],
                'AgriculturalPractices': [6.3, 4.9, 7.2],
                'Encroachments': [5.7, 6.2, 4.8],
                'FloodProbability': [0.35, 0.75, 0.45],
                'FloodRisk': [0, 2, 1]
            }
            df = pd.DataFrame(data)
            df.to_csv(cls.TEST_CSV_PATH, index=False)
    
    def test_predict_endpoint(self):
        """Test the /predict endpoint"""
        test_data = {
            "MonsoonIntensity": 6.5,
            "TopographyDrainage": 5.2,
            "RiverManagement": 4.8,
            "Deforestation": 7.1,
            "Urbanization": 5.3,
            "ClimateChange": 6.2,
            "Siltation": 4.7,
            "AgriculturalPractices": 5.9,
            "Encroachments": 5.4
        }
        
        start_time = time()
        response = requests.post(f"{self.BASE_URL}/predict", json=test_data)
        elapsed_time = time() - start_time
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn('prediction', data)
        self.assertIn('probabilities', data)
        self.assertIn('Low', data['probabilities'])
        self.assertIn('Medium', data['probabilities'])
        self.assertIn('High', data['probabilities'])
        
        print(f"\n/predict response time: {elapsed_time:.3f}s")
    
    def test_upload_endpoint(self):
        """Test the /upload endpoint"""
        with open(self.TEST_CSV_PATH, 'rb') as f:
            files = {'file': ('test_data.csv', f)}
            start_time = time()
            response = requests.post(f"{self.BASE_URL}/upload", files=files)
            elapsed_time = time() - start_time
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        
        print(f"\n/upload response time: {elapsed_time:.3f}s")
    
    def test_retrain_endpoint(self):
        """Test the /retrain endpoint"""
        start_time = time()
        response = requests.post(f"{self.BASE_URL}/retrain")
        elapsed_time = time() - start_time
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('accuracy', data)
        
        print(f"\n/retrain response time: {elapsed_time:.3f}s")
        print(f"Retrained model accuracy: {data['accuracy']:.2f}")
    
    def test_monitor_endpoint(self):
        """Test the /monitor/data endpoint"""
        start_time = time()
        response = requests.get(f"{self.BASE_URL}/monitor/data")
        elapsed_time = time() - start_time
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('accuracy_history', data)
        self.assertIn('response_times', data)
        self.assertIn('prediction_distribution', data)
        
        print(f"\n/monitor/data response time: {elapsed_time:.3f}s")

if __name__ == '__main__':
    unittest.main()
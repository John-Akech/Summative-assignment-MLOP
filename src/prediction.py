import numpy as np
import pandas as pd
import joblib

def predict_flood_risk(input_data, model_path, scaler_path):
    """
    Make flood risk predictions using the trained model.
    
    Args:
        input_data (dict or pd.DataFrame): Input features for prediction
        model_path (str): Path to the trained model
        scaler_path (str): Path to the feature scaler
        
    Returns:
        dict: Prediction results with probabilities
    """
    # Load model and scaler
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    scaler = joblib.load(scaler_path)
    
    # Convert input to DataFrame if it's a dictionary
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    # Ensure we have all expected features
    expected_features = [
        'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
        'Deforestation', 'Urbanization', 'ClimateChange',
        'Siltation', 'AgriculturalPractices', 'Encroachments'
    ]
    
    # Add any missing features with default values
    for feature in expected_features:
        if feature not in input_data.columns:
            input_data[feature] = 0.0  # Default value
    
    # Reorder columns to match training data
    input_data = input_data[expected_features]
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    probabilities = model.predict(input_scaled)
    predicted_class = np.argmax(probabilities, axis=1)
    class_labels = ['Low', 'Medium', 'High']
    prediction = class_labels[predicted_class[0]]
    
    # Prepare response
    response = {
        'prediction': prediction,
        'probabilities': {
            'Low': float(probabilities[0][0]),
            'Medium': float(probabilities[0][1]),
            'High': float(probabilities[0][2])
        }
    }
    
    return response
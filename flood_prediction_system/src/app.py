import os
import time
import psutil
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import json
import traceback
import logging
from logging.handlers import RotatingFileHandler
from flask_wtf.csrf import CSRFProtect, CSRFError
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, FileField
from wtforms.validators import DataRequired, NumberRange
from functools import lru_cache

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Configure CSRF protection
csrf = CSRFProtect(app)
app.config['WTF_CSRF_CHECK_DEFAULT'] = False
app.config['WTF_CSRF_ENABLED'] = True

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
UPLOAD_FOLDER = os.path.join(DATA_FOLDER, 'uploads')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
ALLOWED_EXTENSIONS = {'csv'}
MODEL_PATH = os.path.join(MODEL_FOLDER, 'flood_risk.pkl')
SCALER_PATH = os.path.join(MODEL_FOLDER, 'scaler.pkl')
MONITORING_FILE = os.path.join(BASE_DIR, 'monitoring_data.json')
LOG_FILE = os.path.join(BASE_DIR, 'flood_risk_app.log')
DEFAULT_TRAIN_DATA = os.path.join(DATA_FOLDER, 'train_data.csv')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=1000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize monitoring data structure
def init_monitoring_data():
    return {
        "accuracy_history": [],
        "response_times": {
            "predict": [],
            "upload": [],
            "retrain": [],
            "monitor": [],
            "health": []
        },
        "prediction_distribution": {
            "low": 0,
            "medium": 0,
            "high": 0
        },
        "last_retraining": None,
        "system_status": {
            "model": "Unknown",
            "api": "Unknown",
            "database": "Unknown"
        }
    }

# Global variables for monitoring
monitoring_data = init_monitoring_data()

# Expected features for the model
EXPECTED_FEATURES = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange',
    'Siltation', 'AgriculturalPractices', 'Encroachments'
]

# Load model and scaler at startup with caching
@lru_cache(maxsize=1)
def load_model():
    try:
        logger.info("Loading model...")
        if not os.path.exists(MODEL_PATH):
            logger.warning("Model file not found")
            return None
            
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        monitoring_data["system_status"]["model"] = "Healthy"
        return model
    except Exception as e:
        logger.error(f"Error loading model: {traceback.format_exc()}")
        monitoring_data["system_status"]["model"] = "Unhealthy"
        return None

@lru_cache(maxsize=1)
def load_scaler():
    try:
        logger.info("Loading scaler...")
        if not os.path.exists(SCALER_PATH):
            logger.warning("Scaler file not found")
            return None
            
        scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded successfully")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler: {traceback.format_exc()}")
        return None

# Flask-WTF Forms
class UploadForm(FlaskForm):
    file = FileField('CSV File', validators=[DataRequired()])
    submit = SubmitField('Upload')

class PredictionForm(FlaskForm):
    monsoonIntensity = FloatField('Monsoon Intensity (0-10)', validators=[DataRequired(), NumberRange(min=0, max=10)])
    topographyDrainage = FloatField('Topography Drainage (0-10)', validators=[DataRequired(), NumberRange(min=0, max=10)])
    riverManagement = FloatField('River Management (0-10)', validators=[DataRequired(), NumberRange(min=0, max=10)])
    deforestation = FloatField('Deforestation (0-10)', validators=[DataRequired(), NumberRange(min=0, max=10)])
    urbanization = FloatField('Urbanization (0-10)', validators=[DataRequired(), NumberRange(min=0, max=10)])
    climateChange = FloatField('Climate Change (0-10)', validators=[DataRequired(), NumberRange(min=0, max=10)])
    siltation = FloatField('Siltation (0-10)', validators=[DataRequired(), NumberRange(min=0, max=10)])
    agriculturalPractices = FloatField('Agricultural Practices (0-10)', validators=[DataRequired(), NumberRange(min=0, max=10)])
    encroachments = FloatField('Encroachments (0-10)', validators=[DataRequired(), NumberRange(min=0, max=10)])
    submit = SubmitField('Predict')

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_monitoring_data():
    try:
        with open(MONITORING_FILE, 'w') as f:
            json.dump(monitoring_data, f, indent=2)
        monitoring_data["system_status"]["database"] = "Connected"
    except Exception as e:
        logger.error(f"Error saving monitoring data: {str(e)}")
        monitoring_data["system_status"]["database"] = "Disconnected"

def load_monitoring_data():
    global monitoring_data
    try:
        if os.path.exists(MONITORING_FILE):
            with open(MONITORING_FILE, 'r') as f:
                loaded_data = json.load(f)
                # Start with fresh initialized data
                monitoring_data = init_monitoring_data()
                # Carefully update with loaded data
                for key in loaded_data:
                    if key in monitoring_data:
                        if isinstance(monitoring_data[key], dict):
                            monitoring_data[key].update(loaded_data[key])
                        else:
                            monitoring_data[key] = loaded_data[key]
                
                # Ensure all endpoints exist in response_times
                required_endpoints = ["predict", "upload", "retrain", "monitor", "health"]
                for endpoint in required_endpoints:
                    if endpoint not in monitoring_data["response_times"]:
                        monitoring_data["response_times"][endpoint] = []
                        
        else:
            monitoring_data = init_monitoring_data()
            
        monitoring_data["system_status"]["database"] = "Connected" if os.path.exists(MONITORING_FILE) else "Disconnected"
        
    except Exception as e:
        logger.error(f"Error loading monitoring data: {str(e)}")
        monitoring_data = init_monitoring_data()
        monitoring_data["system_status"]["database"] = "Disconnected"

def create_sample_dataset():
    """Create a sample dataset if none exists"""
    try:
        np.random.seed(42)
        num_samples = 500
        
        data = {
            'MonsoonIntensity': np.random.uniform(0, 10, num_samples),
            'TopographyDrainage': np.random.uniform(0, 10, num_samples),
            'RiverManagement': np.random.uniform(0, 10, num_samples),
            'Deforestation': np.random.uniform(0, 10, num_samples),
            'Urbanization': np.random.uniform(0, 10, num_samples),
            'ClimateChange': np.random.uniform(0, 10, num_samples),
            'Siltation': np.random.uniform(0, 10, num_samples),
            'AgriculturalPractices': np.random.uniform(0, 10, num_samples),
            'Encroachments': np.random.uniform(0, 10, num_samples),
            'FloodProbability': np.random.uniform(0, 1, num_samples),
            'FloodRisk': np.random.choice([0, 1, 2], num_samples, p=[0.6, 0.3, 0.1])
        }
        
        df = pd.DataFrame(data)
        df.to_csv(DEFAULT_TRAIN_DATA, index=False)
        logger.info(f"Created sample dataset at {DEFAULT_TRAIN_DATA}")
        
    except Exception as e:
        logger.error(f"Error creating sample dataset: {traceback.format_exc()}")
        raise

def train_initial_model():
    """Train initial model with default data"""
    try:
        if not os.path.exists(DEFAULT_TRAIN_DATA):
            create_sample_dataset()
        
        df = pd.read_csv(DEFAULT_TRAIN_DATA)
        
        required_columns = EXPECTED_FEATURES + ['FloodProbability', 'FloodRisk']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        X = df[EXPECTED_FEATURES]
        y = df['FloodRisk']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = Sequential([
            Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        optimizer = Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)
        
        y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        joblib.dump(scaler, SCALER_PATH)
        
        monitoring_data["accuracy_history"].append({
            "version": len(monitoring_data["accuracy_history"]) + 1,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat()
        })
        monitoring_data["last_retraining"] = datetime.now().isoformat()
        monitoring_data["system_status"]["model"] = "Healthy"
        save_monitoring_data()
        
        logger.info(f"Initial model trained with accuracy: {accuracy:.2f}")
        
        load_model.cache_clear()
        load_scaler.cache_clear()
    except Exception as e:
        logger.error(f"Error training initial model: {traceback.format_exc()}")
        monitoring_data["system_status"]["model"] = "Unhealthy"
        raise

def init_model():
    """Initialize or load model and scaler"""
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            logger.info("Initializing new model and scaler...")
            train_initial_model()
        load_model()
        load_scaler()
        monitoring_data["system_status"]["api"] = "Operational"
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        monitoring_data["system_status"]["model"] = "Unhealthy"
        raise

def validate_model_and_scaler():
    """Check if model and scaler exist and are valid"""
    model = load_model()
    scaler = load_scaler()
    if model is None or scaler is None:
        monitoring_data["system_status"]["model"] = "Unhealthy"
        raise ValueError("Model or scaler not loaded. Please retrain the model.")
    return model, scaler

# Initialize the model at startup
init_model()
load_monitoring_data()

# Error handler for CSRF
@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    logger.warning(f"CSRF Error: {str(e)}")
    return jsonify({
        'error': 'CSRF token missing or invalid',
        'description': str(e.description)
    }), 400

# ======================
# ROUTES
# ======================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Comprehensive health check endpoint for monitoring"""
    start_time = time.time()
    try:
        # Verify model and scaler are loaded
        model, scaler = validate_model_and_scaler()
        
        # System resource checks
        checks = {
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None,
            "disk_space": psutil.disk_usage('/').free > 100 * 1024 * 1024,  # 100MB minimum
            "memory_available": psutil.virtual_memory().available > 100 * 1024 * 1024,  # 100MB
            "cpu_usage": psutil.cpu_percent() < 90,
            "upload_folder_writable": os.access(app.config['UPLOAD_FOLDER'], os.W_OK),
            "model_folder_writable": os.access(MODEL_FOLDER, os.W_OK),
            "last_retraining": monitoring_data.get("last_retraining"),
            "current_accuracy": monitoring_data["accuracy_history"][-1]["accuracy"] if monitoring_data["accuracy_history"] else None,
            "database_connected": monitoring_data["system_status"]["database"] == "Connected"
        }
        
        status = "healthy" if all(checks.values()) else "degraded"
        
        response = {
            "status": status,
            "version": "1.0.0",
            "checks": checks,
            "system_status": monitoring_data["system_status"],
            "timestamp": datetime.now().isoformat(),
            "response_time": f"{(time.time() - start_time) * 1000:.2f}ms"
        }
        
        monitoring_data["response_times"]["health"].append((time.time() - start_time) * 1000)
        monitoring_data["system_status"]["api"] = "Operational"
        save_monitoring_data()
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Health check failed: {traceback.format_exc()}")
        monitoring_data["system_status"]["api"] = "Degraded"
        save_monitoring_data()
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/predict', methods=['GET', 'POST'])
@csrf.exempt
def predict():
    start_time = time.time()
    try:
        if request.method == 'GET':
            return redirect(url_for('predict_form'))
            
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        
        required_fields = [
            'monsoonIntensity', 'topographyDrainage', 'riverManagement',
            'deforestation', 'urbanization', 'climateChange',
            'siltation', 'agriculturalPractices', 'encroachments'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        input_data = {}
        for field in required_fields:
            try:
                value = float(data[field])
                if not (0 <= value <= 10):
                    return jsonify({
                        'error': f'Field {field} must be between 0 and 10'
                    }), 400
                input_data[field] = value
            except (ValueError, TypeError):
                return jsonify({
                    'error': f'Invalid value for {field}. Must be a number.'
                }), 400
        
        model, scaler = validate_model_and_scaler()
        
        model_input = {
            'MonsoonIntensity': input_data['monsoonIntensity'],
            'TopographyDrainage': input_data['topographyDrainage'],
            'RiverManagement': input_data['riverManagement'],
            'Deforestation': input_data['deforestation'],
            'Urbanization': input_data['urbanization'],
            'ClimateChange': input_data['climateChange'],
            'Siltation': input_data['siltation'],
            'AgriculturalPractices': input_data['agriculturalPractices'],
            'Encroachments': input_data['encroachments']
        }
        
        input_df = pd.DataFrame([model_input])
        input_scaled = scaler.transform(input_df[EXPECTED_FEATURES])
        
        pred_probs = model.predict(input_scaled, verbose=0)[0]
        predicted_class = np.argmax(pred_probs)
        class_labels = ['Low', 'Medium', 'High']
        prediction = class_labels[predicted_class]
        
        probabilities = {
            'Low': float(pred_probs[0]),
            'Medium': float(pred_probs[1]),
            'High': float(pred_probs[2])
        }
        
        monitoring_data["prediction_distribution"][prediction.lower()] += 1
        monitoring_data["response_times"]["predict"].append((time.time() - start_time) * 1000)
        monitoring_data["system_status"]["api"] = "Operational"
        save_monitoring_data()
        
        return jsonify({
            'prediction': prediction,
            'probabilities': probabilities,
            'response_time': f"{(time.time() - start_time) * 1000:.2f}ms"
        })
        
    except Exception as e:
        logger.error(f"Prediction failed: {traceback.format_exc()}")
        monitoring_data["system_status"]["api"] = "Degraded"
        save_monitoring_data()
        return jsonify({'error': str(e)}), 500

@app.route('/predict-form', methods=['GET', 'POST'])
def predict_form():
    form = PredictionForm()
    prediction_result = None
    
    if form.validate_on_submit():
        try:
            data = {
                'monsoonIntensity': form.monsoonIntensity.data,
                'topographyDrainage': form.topographyDrainage.data,
                'riverManagement': form.riverManagement.data,
                'deforestation': form.deforestation.data,
                'urbanization': form.urbanization.data,
                'climateChange': form.climateChange.data,
                'siltation': form.siltation.data,
                'agriculturalPractices': form.agriculturalPractices.data,
                'encroachments': form.encroachments.data
            }
            
            model, scaler = validate_model_and_scaler()
            
            model_input = {
                'MonsoonIntensity': data['monsoonIntensity'],
                'TopographyDrainage': data['topographyDrainage'],
                'RiverManagement': data['riverManagement'],
                'Deforestation': data['deforestation'],
                'Urbanization': data['urbanization'],
                'ClimateChange': data['climateChange'],
                'Siltation': data['siltation'],
                'AgriculturalPractices': data['agriculturalPractices'],
                'Encroachments': data['encroachments']
            }
            
            input_df = pd.DataFrame([model_input])
            input_scaled = scaler.transform(input_df[EXPECTED_FEATURES])
            
            pred_probs = model.predict(input_scaled, verbose=0)[0]
            predicted_class = np.argmax(pred_probs)
            class_labels = ['Low', 'Medium', 'High']
            prediction = class_labels[predicted_class]
            
            prediction_result = {
                'prediction': prediction,
                'probabilities': {
                    'Low': float(pred_probs[0]),
                    'Medium': float(pred_probs[1]),
                    'High': float(pred_probs[2])
                }
            }
            
            monitoring_data["prediction_distribution"][prediction.lower()] += 1
            monitoring_data["system_status"]["api"] = "Operational"
            save_monitoring_data()
            
        except Exception as e:
            logger.error(f"Form prediction failed: {traceback.format_exc()}")
            monitoring_data["system_status"]["api"] = "Degraded"
            save_monitoring_data()
            return render_template('predict.html', form=form, error=str(e))
    
    return render_template('predict.html', form=form, result=prediction_result)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    
    if request.method == 'GET':
        return render_template('upload.html', form=form)
    
    if form.validate_on_submit():
        start_time = time.time()
        try:
            file = form.file.data
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                monitoring_data["response_times"]["upload"].append((time.time() - start_time) * 1000)
                monitoring_data["system_status"]["api"] = "Operational"
                save_monitoring_data()
                
                return jsonify({
                    'success': True,
                    'message': 'File uploaded successfully!',
                    'filename': filename
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Only CSV files are allowed'
                }), 400
                
        except Exception as e:
            logger.error(f"Upload error: {traceback.format_exc()}")
            monitoring_data["system_status"]["api"] = "Degraded"
            save_monitoring_data()
            return jsonify({
                'success': False,
                'message': f'Error processing file: {str(e)}'
            }), 500
    
    return jsonify({
        'success': False,
        'message': 'Invalid form submission'
    }), 400

@app.route('/retrain', methods=['POST'])
@csrf.exempt
def retrain():
    start_time = time.time()
    try:
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            logger.info(f"Created upload directory at {app.config['UPLOAD_FOLDER']}")

        data_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.csv')]
        
        if not data_files:
            logger.warning("No CSV files found for retraining")
            return jsonify({
                'success': False,
                'message': 'No CSV files found in upload folder. Please upload files first.'
            }), 400
        
        dfs = []
        processed_files = 0
        for file in data_files:
            try:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                logger.info(f"Processing file: {file_path}")
                
                encodings = ['utf-8', 'latin1', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError(f"Could not read file {file} with any supported encoding")

                df.columns = df.columns.str.strip().str.lower()
                column_mapping = {
                    'floodrisk': 'FloodRisk',
                    'floodprobability': 'FloodProbability'
                }
                for feature in EXPECTED_FEATURES:
                    column_mapping[feature.lower()] = feature
                
                df = df.rename(columns=lambda x: column_mapping.get(x.lower(), x))
                
                if len(df) < 10:
                    logger.warning(f"File {file} has only {len(df)} rows - skipping")
                    continue
                
                if 'FloodRisk' not in df.columns:
                    logger.info("Generating FloodRisk column")
                    df['FloodRisk'] = np.random.choice([0, 1, 2], size=len(df), p=[0.6, 0.3, 0.1])
                
                for col in EXPECTED_FEATURES:
                    if col not in df.columns:
                        logger.info(f"Generating {col} column")
                        df[col] = np.random.uniform(0, 10, len(df))
                
                if 'FloodProbability' not in df.columns:
                    logger.info("Generating FloodProbability column")
                    if 'FloodRisk' in df.columns:
                        df['FloodProbability'] = df['FloodRisk'].map({0: 0.2, 1: 0.5, 2: 0.8})
                    else:
                        df['FloodProbability'] = np.random.uniform(0, 1, len(df))
                
                for col in EXPECTED_FEATURES + ['FloodProbability', 'FloodRisk']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col] = df[col].fillna(np.random.uniform(0, 10) if col in EXPECTED_FEATURES else 
                                          np.random.uniform(0, 1) if col == 'FloodProbability' else
                                          np.random.choice([0, 1, 2]))
                
                dfs.append(df)
                processed_files += 1
                logger.info(f"Successfully processed {file}")

            except Exception as e:
                logger.error(f"Error processing file {file}: {traceback.format_exc()}")
                continue

        if not dfs:
            logger.error("No valid data could be processed from any files")
            return jsonify({
                'success': False,
                'message': 'No valid data could be processed from uploaded files.'
            }), 400

        logger.info(f"Successfully processed {processed_files}/{len(data_files)} files")
        combined_df = pd.concat(dfs, ignore_index=True)
        
        required_columns = EXPECTED_FEATURES + ['FloodRisk']
        missing_columns = [col for col in required_columns if col not in combined_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns after processing: {missing_columns}")
            return jsonify({
                'success': False,
                'message': f'System error: Could not generate required columns: {", ".join(missing_columns)}'
            }), 500

        X = combined_df[EXPECTED_FEATURES]
        y = combined_df['FloodRisk']
        
        if len(X) < 20:
            logger.error(f"Insufficient data for training: only {len(X)} samples")
            return jsonify({
                'success': False,
                'message': f'Insufficient data for training (minimum 20 samples needed, got {len(X)})'
            }), 400

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        except Exception as e:
            logger.error(f"Scaling failed: {traceback.format_exc()}")
            return jsonify({
                'success': False,
                'message': 'Error scaling data for training'
            }), 500

        try:
            model = Sequential([
                Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dense(16, activation='relu'),
                Dense(3, activation='softmax')
            ])
            
            optimizer = Adam(learning_rate=0.01)
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        except Exception as e:
            logger.error(f"Model creation failed: {traceback.format_exc()}")
            return jsonify({
                'success': False,
                'message': 'Error creating model architecture'
            }), 500

        try:
            history = model.fit(
                X_train_scaled, 
                y_train, 
                epochs=10, 
                batch_size=32, 
                verbose=0,
                validation_split=0.1
            )
            
            y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
            accuracy = accuracy_score(y_test, y_pred)
            
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(model, f)
            joblib.dump(scaler, SCALER_PATH)
            
            monitoring_data["accuracy_history"].append({
                "version": len(monitoring_data["accuracy_history"]) + 1,
                "accuracy": accuracy,
                "timestamp": datetime.now().isoformat(),
                "samples_used": len(combined_df)
            })
            monitoring_data["last_retraining"] = datetime.now().isoformat()
            monitoring_data["response_times"]["retrain"].append((time.time() - start_time) * 1000)
            monitoring_data["system_status"]["model"] = "Healthy"
            monitoring_data["system_status"]["api"] = "Operational"
            save_monitoring_data()
            
            load_model.cache_clear()
            load_scaler.cache_clear()
            
            logger.info(f"Retraining successful! Accuracy: {accuracy:.2f}")
            return jsonify({
                'success': True,
                'accuracy': accuracy,
                'message': f'Model retrained successfully with {len(combined_df)} samples! New accuracy: {accuracy:.2f}',
                'samples_used': len(combined_df),
                'training_time': f"{(time.time() - start_time):.2f} seconds"
            })
            
        except Exception as e:
            logger.error(f"Training failed: {traceback.format_exc()}")
            monitoring_data["system_status"]["model"] = "Unhealthy"
            monitoring_data["system_status"]["api"] = "Degraded"
            save_monitoring_data()
            return jsonify({
                'success': False,
                'message': 'Error during model training'
            }), 500

    except Exception as e:
        logger.error(f"Retraining failed completely: {traceback.format_exc()}")
        monitoring_data["system_status"]["model"] = "Unhealthy"
        monitoring_data["system_status"]["api"] = "Degraded"
        save_monitoring_data()
        return jsonify({
            'success': False,
            'message': 'Unexpected error during retraining process'
        }), 500
        
@app.route('/monitor')
def monitor():
    return render_template('monitor.html')

@app.route('/monitor/data')
def monitor_data():
    start_time = time.time()
    try:
        avg_response_times = {
            "predict": np.mean(monitoring_data["response_times"]["predict"]) if monitoring_data["response_times"]["predict"] else 0,
            "upload": np.mean(monitoring_data["response_times"]["upload"]) if monitoring_data["response_times"]["upload"] else 0,
            "retrain": np.mean(monitoring_data["response_times"]["retrain"]) if monitoring_data["response_times"]["retrain"] else 0,
            "monitor": np.mean(monitoring_data["response_times"]["monitor"]) if monitoring_data["response_times"]["monitor"] else 0
        }
        
        monitoring_data["response_times"]["monitor"].append((time.time() - start_time) * 1000)
        save_monitoring_data()
        
        return jsonify({
            "accuracy_history": monitoring_data["accuracy_history"],
            "response_times": avg_response_times,
            "prediction_distribution": monitoring_data["prediction_distribution"],
            "last_retraining": monitoring_data["last_retraining"],
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Monitor data failed: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

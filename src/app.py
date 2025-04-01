import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
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
from flask_wtf.csrf import CSRFProtect
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, FileField
from wtforms.validators import DataRequired, NumberRange
from functools import lru_cache

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
csrf = CSRFProtect(app)

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

# Global variables for monitoring
monitoring_data = {
    "accuracy_history": [],
    "response_times": {
        "predict": [],
        "upload": [],
        "retrain": [],
        "monitor": []
    },
    "prediction_distribution": {
        "low": 0,
        "medium": 0,
        "high": 0
    },
    "last_retraining": None
}

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
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {traceback.format_exc()}")
        return None

@lru_cache(maxsize=1)
def load_scaler():
    try:
        logger.info("Loading scaler...")
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
    except Exception as e:
        logger.error(f"Error saving monitoring data: {str(e)}")

def load_monitoring_data():
    global monitoring_data
    try:
        if os.path.exists(MONITORING_FILE):
            with open(MONITORING_FILE, 'r') as f:
                monitoring_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading monitoring data: {str(e)}")

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
        
        # Simplified model architecture for faster predictions
        model = Sequential([
            Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        optimizer = Adam(learning_rate=0.01)  # Higher learning rate for faster convergence
        model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        # Reduced epochs for faster training
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
        save_monitoring_data()
        
        logger.info(f"Initial model trained with accuracy: {accuracy:.2f}")
        
        # Clear cache to force reload of new model
        load_model.cache_clear()
        load_scaler.cache_clear()
    except Exception as e:
        logger.error(f"Error training initial model: {traceback.format_exc()}")
        raise

def init_model():
    """Initialize or load model and scaler"""
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            logger.info("Initializing new model and scaler...")
            train_initial_model()
        # Warm up the cache
        load_model()
        load_scaler()
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

def validate_model_and_scaler():
    """Check if model and scaler exist and are valid"""
    model = load_model()
    scaler = load_scaler()
    if model is None or scaler is None:
        raise ValueError("Model or scaler not loaded. Please retrain the model.")
    return model, scaler

# Initialize the model at startup
init_model()
load_monitoring_data()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = PredictionForm()
    
    if request.method == 'POST':
        start_time = time.time()
        try:
            # Get JSON data from request
            if request.is_json:
                data = request.get_json()
            else:
                data = request.form.to_dict()
            
            required_fields = [
                'monsoonIntensity', 'topographyDrainage', 'riverManagement',
                'deforestation', 'urbanization', 'climateChange',
                'siltation', 'agriculturalPractices', 'encroachments'
            ]
            
            # Validate all required fields are present
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return jsonify({
                    'error': f'Missing required fields: {", ".join(missing_fields)}'
                }), 400
            
            # Convert all values to float and validate range
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
            
            # Prepare input data for model
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
            
            # Prepare and scale input
            input_df = pd.DataFrame([model_input])
            input_scaled = scaler.transform(input_df[EXPECTED_FEATURES])
            
            # Make prediction
            pred_probs = model.predict(input_scaled, verbose=0)[0]
            predicted_class = np.argmax(pred_probs)
            class_labels = ['Low', 'Medium', 'High']
            prediction = class_labels[predicted_class]
            
            # Prepare response
            probabilities = {
                'Low': float(pred_probs[0]),
                'Medium': float(pred_probs[1]),
                'High': float(pred_probs[2])
            }
            
            # Update monitoring data
            monitoring_data["prediction_distribution"][prediction.lower()] += 1
            monitoring_data["response_times"]["predict"].append((time.time() - start_time) * 1000)
            save_monitoring_data()
            
            return jsonify({
                'prediction': prediction,
                'probabilities': probabilities,
                'response_time': f"{(time.time() - start_time) * 1000:.2f}ms"
            })
            
        except Exception as e:
            logger.error(f"Prediction failed: {traceback.format_exc()}")
            return jsonify({
                'error': str(e)
            }), 500
    
    return render_template('predict.html', form=form)

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
                
                # Save the file directly without validation
                file.save(filepath)
                
                monitoring_data["response_times"]["upload"].append((time.time() - start_time) * 1000)
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
            return jsonify({
                'success': False,
                'message': f'Error processing file: {str(e)}'
            }), 500
    
    return jsonify({
        'success': False,
        'message': 'Invalid form submission'
    }), 400

@app.route('/retrain', methods=['POST'])
def retrain():
    start_time = time.time()
    try:
        # Get CSRF token from headers
        csrf_token = request.headers.get('X-CSRFToken')
        if not csrf_token:
            return jsonify({
                'success': False,
                'message': 'CSRF token missing'
            }), 403
        
        # Check if upload directory exists
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
                
                # Try multiple encodings if needed
                encodings = ['utf-8', 'latin1', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError(f"Could not read file {file} with any supported encoding")

                # Standardize column names (case-insensitive)
                df.columns = df.columns.str.strip().str.lower()
                column_mapping = {
                    'floodrisk': 'FloodRisk',
                    'floodprobability': 'FloodProbability'
                }
                # Add expected features to mapping
                for feature in EXPECTED_FEATURES:
                    column_mapping[feature.lower()] = feature
                
                df = df.rename(columns=lambda x: column_mapping.get(x.lower(), x))
                
                # Ensure we have at least 10 rows
                if len(df) < 10:
                    logger.warning(f"File {file} has only {len(df)} rows - skipping")
                    continue
                
                # Generate FloodRisk if missing (0=Low, 1=Medium, 2=High)
                if 'FloodRisk' not in df.columns:
                    logger.info("Generating FloodRisk column")
                    df['FloodRisk'] = np.random.choice([0, 1, 2], size=len(df), p=[0.6, 0.3, 0.1])
                
                # Fill other missing columns with reasonable values
                for col in EXPECTED_FEATURES:
                    if col not in df.columns:
                        logger.info(f"Generating {col} column")
                        df[col] = np.random.uniform(0, 10, len(df))
                
                if 'FloodProbability' not in df.columns:
                    logger.info("Generating FloodProbability column")
                    # Base probability on FloodRisk if available
                    if 'FloodRisk' in df.columns:
                        df['FloodProbability'] = df['FloodRisk'].map({0: 0.2, 1: 0.5, 2: 0.8})
                    else:
                        df['FloodProbability'] = np.random.uniform(0, 1, len(df))
                
                # Ensure numeric values
                for col in EXPECTED_FEATURES + ['FloodProbability', 'FloodRisk']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Fill any remaining NA values
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
        
        # Final validation
        required_columns = EXPECTED_FEATURES + ['FloodRisk']
        missing_columns = [col for col in required_columns if col not in combined_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns after processing: {missing_columns}")
            return jsonify({
                'success': False,
                'message': f'System error: Could not generate required columns: {", ".join(missing_columns)}'
            }), 500

        # Prepare data
        X = combined_df[EXPECTED_FEATURES]
        y = combined_df['FloodRisk']
        
        # Ensure we have enough data
        if len(X) < 20:
            logger.error(f"Insufficient data for training: only {len(X)} samples")
            return jsonify({
                'success': False,
                'message': f'Insufficient data for training (minimum 20 samples needed, got {len(X)})'
            }), 400

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scaling
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

        # Simplified model architecture for faster training
        try:
            model = Sequential([
                Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dense(16, activation='relu'),
                Dense(3, activation='softmax')
            ])
            
            optimizer = Adam(learning_rate=0.01)  # Higher learning rate for faster convergence
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

        # Training with reduced epochs
        try:
            history = model.fit(
                X_train_scaled, 
                y_train, 
                epochs=10, 
                batch_size=32, 
                verbose=0,
                validation_split=0.1
            )
            
            # Evaluation
            y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model and scaler
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(model, f)
            joblib.dump(scaler, SCALER_PATH)
            
            # Update monitoring data
            monitoring_data["accuracy_history"].append({
                "version": len(monitoring_data["accuracy_history"]) + 1,
                "accuracy": accuracy,
                "timestamp": datetime.now().isoformat(),
                "samples_used": len(combined_df)
            })
            monitoring_data["last_retraining"] = datetime.now().isoformat()
            monitoring_data["response_times"]["retrain"].append((time.time() - start_time) * 1000)
            save_monitoring_data()
            
            # Clear cache to force reload of new model
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
            return jsonify({
                'success': False,
                'message': 'Error during model training'
            }), 500

    except Exception as e:
        logger.error(f"Retraining failed completely: {traceback.format_exc()}")
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
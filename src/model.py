import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score
import pickle

def build_model(input_shape):
    """
    Build and compile the neural network model.
    
    Args:
        input_shape (tuple): Shape of the input features
        
    Returns:
        Sequential: Compiled Keras model
    """
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=input_shape),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(6, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model

def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
    """
    Train the neural network model.
    
    Args:
        model (Sequential): Compiled Keras model
        X_train (np.array): Training features
        y_train (np.array): Training labels
        X_val (np.array): Validation features (optional)
        y_val (np.array): Validation labels (optional)
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        
    Returns:
        History: Training history object
    """
    validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
    
    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.
    
    Args:
        model (Sequential): Trained Keras model
        X_test (np.array): Test features
        y_test (np.array): Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Evaluate loss and accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'predictions': y_pred
    }

def save_model(model, filepath):
    """
    Save the trained model to a file.
    
    Args:
        model (Sequential): Trained Keras model
        filepath (str): Path to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """
    Load a trained model from a file.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Sequential: Loaded Keras model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model
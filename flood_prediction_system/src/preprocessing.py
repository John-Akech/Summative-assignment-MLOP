import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def preprocess_data(df):
    """
    Preprocess the raw flood risk data.
    
    Args:
        df (pd.DataFrame): Raw input dataframe
        
    Returns:
        pd.DataFrame: Processed dataframe ready for modeling
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # 1. Handle missing values
    processed_df = handle_missing_values(processed_df)
    
    # 2. Remove duplicates
    processed_df = processed_df.drop_duplicates()
    
    # 3. Feature engineering
    processed_df = feature_engineering(processed_df)
    
    return processed_df

def handle_missing_values(df):
    """
    Handle missing values using a combination of imputation methods.
    """
    # Separate numeric and categorical columns
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns
    
    # Numeric imputation - first pass with simple imputer
    num_imputer = SimpleImputer(strategy='mean')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    # Categorical imputation - mode
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    # Second pass with iterative imputer for numeric columns
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(), random_state=42)
    df[num_cols] = iterative_imputer.fit_transform(df[num_cols])
    
    return df

def feature_engineering(df):
    """
    Perform feature engineering and transformations.
    """
    # Create FloodRisk categories from FloodProbability
    df['FloodRisk'] = pd.cut(
        df['FloodProbability'],
        bins=[0, 0.4, 0.6, 1],
        labels=['Low', 'Medium', 'High']
    )
    
    # Convert to numerical values
    df['FloodRisk'] = df['FloodRisk'].map({'Low': 0, 'Medium': 1, 'High': 2}).astype(int)
    
    return df

def prepare_training_data(df, test_size=0.2, random_state=42):
    """
    Prepare data for model training by splitting into features and target,
    and scaling the features.
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    # Split features and target
    X = df.drop(['FloodProbability', 'FloodRisk'], axis=1)
    y = df['FloodRisk']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
# core/model.py
import pandas as pd
import numpy as np
import joblib
import os  # <-- Add this
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

MODEL_PATH = "artifacts/iso_forest_pipeline.joblib"
FEATURES_NUM = ['amount', 'txn_count_24h', 'txn_sum_24h', 'txn_mean_24h', 'time_since_last_txn_h', 'unique_counterparties_24h']
FEATURES_CAT = ['txn_type']

def build_pipeline(available_num, available_cat):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), available_num),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), available_cat)
        ],
        remainder='drop'
    )
    model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1, max_samples=50000)
    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

def train_model(df: pd.DataFrame):
    print("Training Isolation Forest...")
    available_num = [f for f in FEATURES_NUM if f in df.columns]
    available_cat = [f for f in FEATURES_CAT if f in df.columns]
    
    X = df[available_num + available_cat].fillna(0)
    
    pipe = build_pipeline(available_num, available_cat)
    pipe.fit(X)
    
    # --- FIX: Create directory before saving ---
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    joblib.dump(pipe, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return pipe

def score_model(df: pd.DataFrame, pipe) -> pd.DataFrame:
    available_num = [f for f in FEATURES_NUM if f in df.columns]
    available_cat = [f for f in FEATURES_CAT if f in df.columns]
    X = df[available_num + available_cat].fillna(0)
    
    raw_scores = pipe.decision_function(X)
    anomaly_score = 1 / (1 + np.exp(raw_scores)) # Sigmoid squashing to 0-1
    
    df['anomaly_score'] = anomaly_score
    df['ml_flag'] = (anomaly_score > 0.7).astype(int)
    return df

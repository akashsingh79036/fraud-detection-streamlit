# core/model.py
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --- Config ---
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "aml_hybrid_pipeline.joblib"

# Features used by BOTH models (Keep consistent!)
FEATURES_NUM = [
    'amount', 'txn_count_24h', 'txn_sum_24h', 'txn_mean_24h', 'txn_std_24h',
    'time_since_last_txn_h', 'unique_counterparties_24h', 'account_age_days',
    'is_dormant_wakeup', 'rule_score'  # <--- Rule score is a POWERFUL feature
]
FEATURES_CAT = ['txn_type']
ALL_FEATURES = FEATURES_NUM + FEATURES_CAT
TARGET_COL = 'is_suspicious'

# ---------------------------------------------------------
# PREPROCESSING (Shared)
# ---------------------------------------------------------
def get_preprocessor(available_num, available_cat):
    """Robust preprocessor: handles missing cols, unknown categories."""
    transformers = []
    if available_num:
        transformers.append(('num', StandardScaler(), available_num))
    if available_cat:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), available_cat))
    
    return ColumnTransformer(transformers=transformers, remainder='drop')

def prepare_xy(df: pd.DataFrame):
    """Aligns dataframe to expected features, fills missing, returns X, y."""
    df = df.copy()
    # Ensure all expected columns exist
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = 0 if c in FEATURES_NUM else 'UNKNOWN'
    
    # Type safety
    for c in FEATURES_NUM: df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in FEATURES_CAT: df[c] = df[c].astype(str).fillna('UNKNOWN')
    
    X = df[ALL_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    y = df[TARGET_COL].astype(int) if TARGET_COL in df.columns else None
    return X, y

# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------
def train_model(df: pd.DataFrame) -> dict:
    print("🛠️ Training Hybrid AML Model (XGBoost + IsolationForest)...")
    X, y = prepare_xy(df)
    
    available_num = [c for c in FEATURES_NUM if c in X.columns]
    available_cat = [c for c in FEATURES_CAT if c in X.columns]
    preprocessor = get_preprocessor(available_num, available_cat)
    
    artifact = {'preprocessor': preprocessor, 'features': ALL_FEATURES, 'threshold': 0.5}

    # ======================================================
    # 1. SUPERVISED: XGBOOST (Learns YOUR 537 Labels)
    # ======================================================
    xgb_model = None
    optimal_thresh = 0.5
    
    if y is not None and y.sum() > 20: # Need minimum labels
        print(f"   🎯 Labels found: {y.sum()} suspicious / {len(y)} total.")
        
        # Split for Threshold Tuning (Stratified!)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Fit Preprocessor on Train Only
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
        
        # Class Weight Calculation
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos = neg / max(pos, 1)
        print(f"   ⚖️ Class Imbalance Ratio (scale_pos_weight): {scale_pos:.1f}")
        
        xgb_model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,       # 🔑 FIXES "All Zeros"
            eval_metric='aucpr',              # 🔑 Optimizes for Precision-Recall
            tree_method='hist',
            n_jobs=-1,
            random_state=42,
            early_stopping_rounds=30
        )
        
        xgb_model.fit(
            X_train_proc, y_train,
            eval_set=[(X_val_proc, y_val)],
            verbose=False
        )
        
        # -------------------------------------------------
        # THRESHOLD TUNING: Target Recall >= 80% (RBI/Compliance Std)
        # -------------------------------------------------
        val_probs = xgb_model.predict_proba(X_val_proc)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
        
        target_recall = 0.80
        # thresholds array is 1 shorter than precision/recall
        valid_idx = np.where(recall[:-1] >= target_recall)[0]
        
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(precision[valid_idx])]
            optimal_thresh = float(thresholds[best_idx])
            print(f"   ✅ Threshold Tuned: {optimal_thresh:.4f} | Recall: {recall[best_idx]:.2%} | Precision: {precision[best_idx]:.2%}")
        else:
            # Fallback: Max F1
            f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
            optimal_thresh = float(thresholds[np.argmax(f1[:-1])])
            print(f"   ⚠️ Target Recall {target_recall} not reached. Using Max F1 Thresh: {optimal_thresh:.4f}")
            
        artifact['xgb'] = xgb_model
    else:
        print("   ⚠️ Insufficient labels. Skipping Supervised Training.")
        # Fit preprocessor on all data if no labels
        preprocessor.fit(X)
        artifact['xgb'] = None

    # ======================================================
    # 2. UNSUPERVISED: ISOLATION FOREST (Novelty Detection)
    # ======================================================
    print("   🌲 Training IsolationForest on Normal Behavior...")
    # Train ISO ONLY on Normal transactions (y==0) if labels exist
    if y is not None:
        X_normal = X[y == 0]
    else:
        X_normal = X
    
    # Subsample for speed (IF scales O(N^2) roughly)
    if len(X_normal) > 100_000:
        X_normal = X_normal.sample(100_000, random_state=42)
    
    X_normal_proc = preprocessor.transform(X_normal)
    
    iso_model = IsolationForest(
        n_estimators=300,
        contamination=0.005,      # Expect ~0.5% anomalies in "normal" pool
        max_samples=min(256, len(X_normal)),
        random_state=42,
        n_jobs=-1
    )
    iso_model.fit(X_normal_proc)
    artifact['iso'] = iso_model
    print(f"   ✅ IsolationForest trained on {len(X_normal_proc)} normal samples.")

    # ======================================================
    # 3. SAVE ARTIFACT
    # ======================================================
    artifact['threshold'] = optimal_thresh
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"   💾 Full Pipeline Saved: {MODEL_PATH}")
    return artifact

# ---------------------------------------------------------
# SCORING (Called by Dashboard)
# ---------------------------------------------------------
def score_model(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    df = df.copy()
    X, _ = prepare_xy(df)
    
    # Align columns strictly to training features
    X = X.reindex(columns=artifact['features'], fill_value=0)
    X_proc = artifact['preprocessor'].transform(X)
    
    final_scores = np.zeros(len(X))
    
    # A. Supervised Score (Probability) - Weight 0.7
    if artifact.get('xgb') is not None:
        xgb_probs = artifact['xgb'].predict_proba(X_proc)[:, 1]
        final_scores += 0.7 * xgb_probs
    
    # B. Unsupervised Score (Anomaly) - Weight 0.3
    # ISO: decision_function > 0 = Normal, < 0 = Anomaly
    # We need 0-1 scale where 1 = Anomaly
    iso_raw = artifact['iso'].decision_function(X_proc)
    # Convert to "Anomaly Probability" via MinMax on the negative side
    # Shift so 0 is approx boundary, squash
    iso_score = 1 / (1 + np.exp(iso_raw * 5)) # Sigmoid scaling factor 5
    final_scores += 0.3 * iso_score
    
    df['anomaly_score'] = np.clip(final_scores, 0, 1)
    
    # C. Binary Flag using OPTIMIZED Threshold
    thresh = artifact.get('threshold', 0.5)
    df['ml_flag'] = (df['anomaly_score'] >= thresh).astype(int)
    
    print(f"   📊 Scoring Complete | Flags: {df['ml_flag'].sum()} | Mean Score: {df['anomaly_score'].mean():.4f} | Thresh: {thresh:.4f}")
    return df

# ---------------------------------------------------------
# LOAD HELPER (For @st.cache_resource)
# ---------------------------------------------------------
def load_or_train(df: pd.DataFrame) -> dict:
    if MODEL_PATH.exists():
        print("   ♻️ Loading Existing Hybrid Model...")
        return joblib.load(MODEL_PATH)
    print("   🆕 No Model Found. Training New...")
    return train_model(df)

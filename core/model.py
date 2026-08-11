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

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "aml_hybrid_pipeline.joblib"

FEATURES_NUM = [
    'amount', 'txn_count_24h', 'txn_sum_24h', 'txn_mean_24h', 'txn_std_24h',
    'time_since_last_txn_h', 'unique_counterparties_24h', 'account_age_days',
    'is_dormant_wakeup', 'rule_score'
]
FEATURES_CAT = ['txn_type']
ALL_FEATURES = FEATURES_NUM + FEATURES_CAT
TARGET_COL = 'is_suspicious'

def get_preprocessor(available_num, available_cat):
    transformers = []
    if available_num: transformers.append(('num', StandardScaler(), available_num))
    if available_cat: transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), available_cat))
    return ColumnTransformer(transformers=transformers, remainder='drop')

def prepare_xy(df: pd.DataFrame):
    df = df.copy()
    for c in ALL_FEATURES:
        if c not in df.columns: df[c] = 0 if c in FEATURES_NUM else 'UNKNOWN'
    for c in FEATURES_NUM: df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in FEATURES_CAT: df[c] = df[c].astype(str).fillna('UNKNOWN')
    X = df[ALL_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    y = df[TARGET_COL].astype(int) if TARGET_COL in df.columns else None
    return X, y

def train_model(df: pd.DataFrame) -> dict:
    print("\n" + "="*50)
    print("🛠️ TRAINING HYBRID AML MODEL")
    print("="*50)
    X, y = prepare_xy(df)
    available_num = [c for c in FEATURES_NUM if c in X.columns]
    available_cat = [c for c in FEATURES_CAT if c in X.columns]
    preprocessor = get_preprocessor(available_num, available_cat)
    artifact = {'preprocessor': preprocessor, 'features': ALL_FEATURES, 'threshold': 0.5}

    # ======================================================
    # 1. SUPERVISED: XGBOOST
    # ======================================================
    xgb_model = None
    optimal_thresh = 0.5
    
    if y is not None and y.sum() > 20:
        print(f"🎯 Labels: {y.sum()} suspicious / {len(y)} total.")
        
        # STRATIFY FALLBACK
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            if y_train.sum() == 0: raise ValueError("Stratify produced 0 positives in train")
        except ValueError:
            print("⚠️ Stratify failed (too few positives), using random split.")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Train Pos: {y_train.sum()}")
        
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
        
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos = neg / max(pos, 1)
        print(f"   ⚖️ Scale Pos Weight: {scale_pos:.1f}")
        
        xgb_model = XGBClassifier(
            n_estimators=800, max_depth=8, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
            reg_lambda=2.0, reg_alpha=0.1,
            scale_pos_weight=scale_pos,
            eval_metric='aucpr', tree_method='hist', n_jobs=-1,
            random_state=42, early_stopping_rounds=50
            # REMOVED monotone_constraints
        )
        
        print("   🏋️ Fitting XGBoost...")
        xgb_model.fit(X_train_proc, y_train, eval_set=[(X_val_proc, y_val)], verbose=True) # VERBOSE TRUE
        
        # THRESHOLD TUNING
        val_probs = xgb_model.predict_proba(X_val_proc)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
        target_recall = 0.80
        valid_idx = np.where(recall[:-1] >= target_recall)[0]
        
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(precision[valid_idx])]
            optimal_thresh = float(thresholds[best_idx])
            print(f"   ✅ Threshold: {optimal_thresh:.4f} | Recall: {recall[best_idx]:.2%} | Prec: {precision[best_idx]:.2%}")
        else:
            f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
            optimal_thresh = float(thresholds[np.argmax(f1[:-1])])
            print(f"   ⚠️ Max F1 Threshold: {optimal_thresh:.4f}")
            
        artifact['xgb'] = xgb_model
    else:
        print("   ⚠️ No labels. Fitting Preprocessor only.")
        preprocessor.fit(X)
        artifact['xgb'] = None

    # ======================================================
    # 2. UNSUPERVISED: ISOLATION FOREST
    # ======================================================
    print("   🌲 Training IsolationForest...")
    X_normal = X[y == 0] if y is not None else X
    if len(X_normal) > 100_000: X_normal = X_normal.sample(100_000, random_state=42)
    X_normal_proc = preprocessor.transform(X_normal)
    
    iso_model = IsolationForest(n_estimators=300, contamination=0.005, max_samples=256, random_state=42, n_jobs=-1)
    iso_model.fit(X_normal_proc)
    artifact['iso'] = iso_model
    print(f"   ✅ ISO trained on {len(X_normal_proc)} samples.")

    artifact['threshold'] = optimal_thresh
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"   💾 Saved: {MODEL_PATH}")
    print("="*50 + "\n")
    return artifact

def score_model(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    df = df.copy()
    X, _ = prepare_xy(df)
    X = X.reindex(columns=artifact['features'], fill_value=0)
    X_proc = artifact['preprocessor'].transform(X)
    
    final_scores = np.zeros(len(X))
    
    if artifact.get('xgb') is not None:
        xgb_probs = artifact['xgb'].predict_proba(X_proc)[:, 1]
        final_scores += 0.7 * xgb_probs
        print(f"   📊 XGB Score Range: {xgb_probs.min():.4f} - {xgb_probs.max():.4f} | Mean: {xgb_probs.mean():.4f}")
    
    iso_raw = artifact['iso'].decision_function(X_proc)
    iso_score = 1 / (1 + np.exp(iso_raw * 5))
    final_scores += 0.3 * iso_score
    
    df['anomaly_score'] = np.clip(final_scores, 0, 1)
    thresh = artifact.get('threshold', 0.5)
    df['ml_flag'] = (df['anomaly_score'] >= thresh).astype(int)
    print(f"   🏁 Final Hybrid Score Mean: {df['anomaly_score'].mean():.4f} | Flags @ {thresh:.4f}: {df['ml_flag'].sum()}")
    return df

def load_or_train(df: pd.DataFrame) -> dict:
    if MODEL_PATH.exists():
        print(f"   ♻️ Loading: {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    print("   🆕 No model found. Training...")
    return train_model(df)

# ======================================================
# EXECUTION BLOCK: RUN TRAINING AND HYBRID SCORING PIPELINE
# ======================================================
if __name__ == "__main__":
    import os

    print("\n--- Pipeline Initialization ---")
    data_file = "data/processed_transactions.csv"
    
    # 1. Double-check that feature engineering file exists
    if not os.path.exists(data_file):
        print(f"❌ Error: {data_file} not found. Please execute 'python core/engine.py' first!")
    else:
        # 2. Read the processed database
        raw_df = pd.read_csv(data_file)
        print(f"Dataset successfully loaded. Total rows to analyze: {len(raw_df)}")
        
        # 3. Fit the complete hybrid architecture (XGBoost + Preprocessor + IsoForest)
        trained_artifact = train_model(raw_df)
        
        print("\n" + "="*50)
        print("🔮 SCORING DATASET WITH HYBRID ML PIPELINE")
        print("="*50)
        
        # 4. Generate final composite anomaly scores and machine learning flags
        scored_output_df = score_model(raw_df, trained_artifact)
        
        # 5. Save the final ML scored dataset
        final_csv_path = "data/ml_final_scored_transactions.csv"
        scored_output_df.to_csv(final_csv_path, index=False)
        
        print(f"📦 Pipeline complete! Machine learning data flags saved to: {final_csv_path}")
        print("="*50 + "\n")

# core/scoring.py
import os
import sys
import pandas as pd
import numpy as np
import joblib

def calculate_final_risk(df: pd.DataFrame, artifact: dict = None) -> pd.DataFrame:
    df = df.copy()
    
    # ---------------------------------------------------------
    # 1. DEFAULTS & CLEANING
    # ---------------------------------------------------------
    for c in ['rule_score', 'anomaly_score']:
        if c not in df.columns: df[c] = 0.0
        df[c] = df[c].fillna(0).clip(lower=0)
    
    # ---------------------------------------------------------
    # 2. ROBUST NORMALIZATION (Rule Score)
    # ---------------------------------------------------------
    rule_cap = df['rule_score'].quantile(0.99)
    if rule_cap < 1: rule_cap = 1.0 # Floor
    
    norm_rule = (df['rule_score'] / rule_cap).clip(0, 1)
    norm_ml = df['anomaly_score'].clip(0, 1) 

    # ---------------------------------------------------------
    # 3. HYBRID CONTINUOUS SCORE (For Ranking/Prioritization)
    # ---------------------------------------------------------
    df['risk_score'] = (0.6 * norm_rule) + (0.4 * norm_ml)

    # ---------------------------------------------------------
    # 4. DISCRETE RISK LEVELS (DECOUPLED LOGIC)
    # ---------------------------------------------------------
    ml_thresh = artifact.get('threshold', 0.5) if artifact else 0.5
    
    rule_hit = df['rule_score'] > 0
    ml_hit = df['anomaly_score'] >= ml_thresh
    
    conditions = [
        rule_hit | ml_hit,                    
        (df['risk_score'] > 0.3) & ~(rule_hit | ml_hit) 
    ]
    choices = ['HIGH', 'MEDIUM']
    df['risk_level'] = np.select(conditions, choices, default='LOW')
    
    df['risk_level'] = pd.Categorical(df['risk_level'], categories=['LOW', 'MEDIUM', 'HIGH'], ordered=True)

    # ---------------------------------------------------------
    # 5. KPI FLAGS
    # ---------------------------------------------------------
    df['rule_flag'] = rule_hit.astype(int)
    df['ml_flag'] = ml_hit.astype(int) 

    # ---------------------------------------------------------
    # 6. INVESTIGATION PRIORITY (Sorting)
    # ---------------------------------------------------------
    risk_weight = df['risk_level'].map({'HIGH': 3.0, 'MEDIUM': 2.0, 'LOW': 1.0}).astype(float)
    rule_bonus = 1 + df['rule_score'].clip(0, 3) 
    ml_bonus = 1 + norm_ml 
    
    df['investigation_priority'] = risk_weight * rule_bonus * ml_bonus

    return df.sort_values('investigation_priority', ascending=False).reset_index(drop=True)


# ======================================================
# EXECUTION BLOCK: RUN LIVE OPERATIONAL RISK RANKING
# ======================================================
if __name__ == "__main__":
    print("\n--- Pipeline Initialization ---")
    input_data_path = "data/ml_final_scored_transactions.csv"
    model_path = "artifacts/aml_hybrid_pipeline.joblib"
    
    # 1. Ensure the scored transactions exist
    if not os.path.exists(input_data_path):
        print(f"❌ Error: {input_data_path} not found. Please run core/model.py first!")
        sys.exit()
        
    df_scored = pd.read_csv(input_data_path)
    print(f"Loaded {len(df_scored)} pre-scored transaction records.")
    
    # 2. Load the trained model artifact to get your calibrated threshold
    trained_artifact = None
    if os.path.exists(model_path):
        print(f"♻️ Loading model configurations from: {model_path}")
        trained_artifact = joblib.load(model_path)
    else:
        print("⚠️ Warning: Model artifact not found. Using default threshold of 0.5.")
        
    print("\n--- Calculating Final Operational Risk Matrix ---")
    # 3. Apply your robust risk calculation matrix
    final_operational_df = calculate_final_risk(df_scored, trained_artifact)
    
    # 4. Print pipeline risk summary metrics
    print("\n=== Investigation Alert Dashboard Metrics ===")
    print(final_operational_df['risk_level'].value_counts().to_string())
    
    # 5. Save out the operational database file
    output_csv_path = "data/operational_alerts_final.csv"
    final_operational_df.to_csv(output_csv_path, index=False)
    print(f"\n📦 Success! Saved finalized operational audit sheet to: {output_csv_path}")

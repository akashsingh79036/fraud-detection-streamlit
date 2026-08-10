# core/scoring.py
import pandas as pd
import numpy as np

def calculate_final_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines Rule Score (0-N) and ML Anomaly Score (0-1) into Final Risk.
    Creates: risk_score, risk_level, investigation_priority, rule_flag.
    """
    df = df.copy()
    
    # ---------------------------------------------------------
    # 1. INPUT VALIDATION & DEFAULTS
    # ---------------------------------------------------------
    if 'rule_score' not in df.columns:
        df['rule_score'] = 0.0
    if 'anomaly_score' not in df.columns:
        df['anomaly_score'] = 0.0
    
    # Fill NaNs just in case
    df['rule_score'] = df['rule_score'].fillna(0).clip(lower=0)
    df['anomaly_score'] = df['anomaly_score'].fillna(0).clip(0, 1)

    # ---------------------------------------------------------
    # 2. NORMALIZATION
    # ---------------------------------------------------------
    # Rule Score: Min-Max (0-1). Max observed in data (dynamic) or fixed cap (e.g., 10).
    # Using dynamic max with a floor of 1 to avoid div/0.
    max_rule = df['rule_score'].max()
    norm_rule = df['rule_score'] / max(max_rule, 1.0)
    
    # ML Score: Already 0-1 from model.py sigmoid
    norm_ml = df['anomaly_score']

    # ---------------------------------------------------------
    # 3. HYBRID RISK SCORE (Weighted Sum)
    # ---------------------------------------------------------
    # Weights: Rules are deterministic/high precision -> 0.6. ML catches novel -> 0.4.
    df['risk_score'] = (0.6 * norm_rule) + (0.4 * norm_ml)
    df['risk_score'] = df['risk_score'].clip(0, 1)

    # ---------------------------------------------------------
    # 4. RISK BUCKETS (Categorical)
    # ---------------------------------------------------------
    bins = [0, 0.30, 0.65, 1.0]
    labels = ['LOW', 'MEDIUM', 'HIGH']
    # include_lowest=True ensures 0.0 maps to LOW
    df['risk_level'] = pd.cut(df['risk_score'], bins=bins, labels=labels, include_lowest=True)
    
    # Explicit Rule Flag for Dashboard KPI (Any rule hit)
    df['rule_flag'] = (df['rule_score'] > 0).astype(int)

    # ---------------------------------------------------------
    # 5. INVESTIGATION PRIORITY (Sorting Key)
    # ---------------------------------------------------------
    # Logic: Priority = Risk_Level_Weight * (1 + Rule_Severity)
    # Risk_Level_Weight: HIGH=3, MEDIUM=2, LOW=1
    # Rule_Severity: capped rule_score (max 3 points bonus)
    
    # SAFE MAPPING: Categorical -> String -> Map -> FillNA -> Float
    # This avoids: TypeError: Cannot multiply Categorical by float
    risk_priority_map = {'HIGH': 3.0, 'MEDIUM': 2.0, 'LOW': 1.0}
    
    mapped_priority = (
        df['risk_level']
        .astype(str)           # Convert Categorical to String
        .map(risk_priority_map) # Map to numeric weights
        .fillna(1.0)           # Fallback for any unexpected labels
        .astype(float)         # Ensure float type for multiplication
    )
    
    # Rule contribution (cap at 3 so rules don't drown risk level)
    rule_contrib = df['rule_score'].clip(0, 3)
    
    df['investigation_priority'] = mapped_priority * (1 + rule_contrib)
    
    # ---------------------------------------------------------
    # 6. FINAL SORT
    # ---------------------------------------------------------
    return df.sort_values('investigation_priority', ascending=False).reset_index(drop=True)

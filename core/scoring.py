# core/scoring.py
import pandas as pd
import numpy as np

def calculate_final_risk(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Normalize Rule Score (Max expected ~5-6 in our rules)
    max_rule = df['rule_score'].max() if df['rule_score'].max() > 0 else 1
    norm_rule = df['rule_score'] / max_rule
    
    # 2. ML Score already 0-1
    norm_ml = df['anomaly_score']
    
    # 3. Hybrid Score
    df['risk_score'] = (0.6 * norm_rule) + (0.4 * norm_ml)
    
    # 4. Risk Buckets (Creates Categorical column)
    bins = [0, 0.3, 0.6, 1.0]
    labels = ['LOW', 'MEDIUM', 'HIGH']
    df['risk_level'] = pd.cut(df['risk_score'], bins=bins, labels=labels, include_lowest=True)
    
    # 5. Investigation Priority -- FIX HERE ---
    # .map() on Categorical returns Categorical. Must cast to numeric before multiplication.
    risk_priority_map = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    
    # Option A: Convert mapped series to float explicitly
    mapped_priority = df['risk_level'].map(risk_priority_map).astype(float)
    
    # Option B (Cleaner): Use the numeric codes from the categorical directly (0,1,2) + 1
    # mapped_priority = df['risk_level'].cat.codes + 1 # LOW=1, MED=2, HIGH=3
    
    df['investigation_priority'] = mapped_priority * (1 + df['rule_score'].clip(0, 3))
    
    return df.sort_values('investigation_priority', ascending=False)

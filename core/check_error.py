# 1. NUKE OLD ARTIFACTS (Critical: filenames changed)
#rm -rf artifacts/

# 2. RUN DEBUG SCRIPT (Copy paste this into terminal)
#python << 'EOF'
import pandas as pd
import sys
import os
sys.path.append(".") 

from core.engine import compute_behavioral_features, apply_rule_engine
from core.model import load_or_train, score_model, FEATURES_NUM, FEATURES_CAT
from core.scoring import calculate_final_risk

print("--- 1. Loading Data ---")
df = pd.read_csv("data/raw/transactions.csv", parse_dates=['timestamp'])
print(f"Raw: {len(df)} rows")

print("--- 2. Features & Rules ---")
df = compute_behavioral_features(df)
df = apply_rule_engine(df)
print(f"Features done. Rule Hits: {df['rule_score'].sum()}")

print("--- 3. Model Load/Train ---")
artifact = load_or_train(df)
print(f"Artifact Keys: {artifact.keys()}")
print(f"  XGB Present: {artifact.get('xgb') is not None}")
print(f"  ISO Present: {artifact.get('iso') is not None}")
print(f"  Threshold: {artifact.get('threshold')}")
print(f"  Features Expected: {artifact.get('features')}")

print("--- 4. Scoring ---")
df = score_model(df, artifact)
print(f"Anomaly Score Stats:\n{df['anomaly_score'].describe()}")
print(f"ML Flags (at threshold {artifact.get('threshold', 0.5)}): {df['ml_flag'].sum()}")
print(f"Score > 0.5: {(df['anomaly_score'] > 0.5).sum()}")
print(f"Score > 0.1: {(df['anomaly_score'] > 0.1).sum()}")
print(f"Score > 0.01: {(df['anomaly_score'] > 0.01).sum()}")

print("--- 5. Final Risk ---")
df = calculate_final_risk(df, artifact)
print(f"HIGH Risk: {(df['risk_level'] == 'HIGH').sum()}")
print(f"ML Flags Final: {df['ml_flag'].sum()}")
print(f"Rule Flags Final: {df['rule_flag'].sum()}")

# Check Ground Truth Overlap
if 'is_suspicious' in df.columns:
    susp = df[df['is_suspicious']==1]
    print(f"\n--- Ground Truth (537) ---")
    print(f"  Caught by Rules: {susp['rule_flag'].sum()}")
    print(f"  Caught by ML:    {susp['ml_flag'].sum()}")
    print(f"  Caught by Either: {((susp['rule_flag']==1) | (susp['ml_flag']==1)).sum()}")
    print(f"  ML Scores for Susp:\n{susp['anomaly_score'].describe()}")
#EOF

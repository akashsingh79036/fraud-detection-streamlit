# core/engine.py
import pandas as pd
import numpy as np

THRESHOLD_REPORTING = 10_00_000
DORMANT_DAYS = 90

def compute_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Sort + Reset Index (Critical for .values alignment)
    df = df.sort_values(['sender_acc', 'timestamp']).reset_index(drop=True)

    # 2. Rolling Aggregations (Single Pass via .agg)
    # Only 'amount' needs numeric stats. 
    rolled_agg = df.groupby('sender_acc', group_keys=False).rolling(
        '24h', on='timestamp', closed='left'
    )[['amount']].agg(['count', 'sum', 'mean', 'std'])
    
    # Assign via .values (Fast, no index alignment)
    df['txn_count_24h'] = rolled_agg[('amount', 'count')].values
    df['txn_sum_24h']   = rolled_agg[('amount', 'sum')].values
    df['txn_mean_24h']  = rolled_agg[('amount', 'mean')].values
    df['txn_std_24h']   = rolled_agg[('amount', 'std')].values

    # 3. Time Since Last Txn (Vectorized Grouped Diff)
    df['time_since_last_txn_h'] = df.groupby('sender_acc')['timestamp'].diff().dt.total_seconds() / 3600

    # 4. Unique Counterparties 24h (O(N) 2-Pointer Sliding Window)
    # Much faster than rolling.apply(nunique)
    unique_counts = np.zeros(len(df), dtype=int)
    
    for _, g in df.groupby('sender_acc', sort=False):
        idx = g.index.values
        times = g['timestamp'].values
        recvs = g['receiver_acc'].values
        
        left = 0
        counts = {}
        for right in range(len(idx)):
            r = recvs[right]
            counts[r] = counts.get(r, 0) + 1
            
            while times[right] - times[left] > np.timedelta64(24, 'h'):
                l_recv = recvs[left]
                counts[l_recv] -= 1
                if counts[l_recv] == 0:
                    del counts[l_recv]
                left += 1
            
            unique_counts[idx[right]] = len(counts)
            
    df['unique_counterparties_24h'] = unique_counts

    # 5. Dormancy Features
    first_txn = df.groupby('sender_acc')['timestamp'].transform('min')
    df['account_age_days'] = (df['timestamp'] - first_txn).dt.total_seconds() / 86400
    df['is_dormant_wakeup'] = (df['time_since_last_txn_h'] > DORMANT_DAYS * 24) & (df['account_age_days'] > DORMANT_DAYS)

    # 6. Fill NaNs (First txn per account)
    feat_cols = ['txn_count_24h', 'txn_sum_24h', 'txn_mean_24h', 'txn_std_24h', 
                 'time_since_last_txn_h', 'unique_counterparties_24h']
    df[feat_cols] = df[feat_cols].fillna(0)
    
    return df

def apply_rule_engine(df: pd.DataFrame) -> pd.DataFrame:
    rules_triggered = [[] for _ in range(len(df))]
    scores = np.zeros(len(df))

    # R1: Structuring
    mask = (df['amount'] > 0.9 * THRESHOLD_REPORTING) & \
           (df['amount'] < THRESHOLD_REPORTING) & \
           (df['txn_count_24h'] >= 3)
    scores[mask] += 2
    for i in np.where(mask)[0]: rules_triggered[i].append("R1_STRUCTURING")

    # R2: High Velocity
    mask = (df['txn_count_24h'] > 10) & (df['txn_mean_24h'] < 50_000)
    scores[mask] += 1
    for i in np.where(mask)[0]: rules_triggered[i].append("R2_HIGH_VELOCITY")

    # R3: Dormant Wakeup
    mask = df['is_dormant_wakeup'] & (df['amount'] > 5_00_000)
    scores[mask] += 3
    for i in np.where(mask)[0]: rules_triggered[i].append("R3_DORMANT_WAKEUP")

    # R4: High Risk Geo
    mask = df['receiver_branch'].astype(str).str.startswith('HR_') | \
           df['sender_branch'].astype(str).str.startswith('HR_')
    scores[mask] += 2
    for i in np.where(mask)[0]: rules_triggered[i].append("R4_HIGH_RISK_GEO")

    # R5: Round Numbers
    mask = (df['amount'] % 1000 == 0) | (df['amount'] % 100000 == 99999)
    scores[mask] += 0.5
    for i in np.where(mask)[0]: rules_triggered[i].append("R5_ROUND_NUM")

    df['rule_score'] = scores
    df['triggered_rules'] = rules_triggered
    return df

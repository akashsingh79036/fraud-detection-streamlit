import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import call_llm

print("Starting data generation process...")

# 1. Ask the AI to create something for your project
prompt_text = "Generate 3 rows of fake website traffic data in JSON format with fields: timestamp, ip_address, and status_code."
ai_response = call_llm(prompt=prompt_text)

# 2. Print or save the results
print("\nGenerated Data from AI:")
print(ai_response)

# Make the AI act like a code cleaner
clean_code = call_llm(
    prompt="Fix syntax errors in this snippet: print 'hello'",
    system_prompt="You are a strict Python code formatting assistant."
)

# Make the AI act like a data analyst
analysis = call_llm(
    prompt="We had a 500% spike in traffic at 2 AM. What could cause this?",
    system_prompt="You are a security analyst looking for server attacks."
)


fake = Faker('en_IN') # Indian locale
N_CUSTOMERS = 5000
N_TXNS = 200_000
OUTPUT_PATH = "data/raw/transactions.csv"

# ---------------------------------------------------------
# OPTION A: Use LLM to generate the *logic* for complex patterns
# (Run once, copy output, paste below, or just use the coded version below)
# ---------------------------------------------------------
PROMPT = """
Write a Python function `generate_aml_dataset(n_customers, n_txns)` using Faker (en_IN).
It must return a pandas DataFrame with columns:
txn_id, timestamp, sender_acc, receiver_acc, sender_cust_id, receiver_cust_id, 
amount, txn_type (NEFT/RTGS/IMPS/UPI/CASH), branch_sender, branch_receiver, 
is_suspicious (0/1), suspicious_pattern (None/'structuring'/'layering'/'dormant_wakeup'/'circular'/'high_risk_country').

Implement these specific patterns programmatically:
1. Structuring: Sender sends multiple txns < 10L (e.g., 9.9L) to same/diff receivers within 24h.
2. Layering: Chain A->B->C->D (3+ hops) within 1 hour, amounts ~similar.
3. Dormant Wakeup: Account 0 txns for 90 days, then sudden >50L movement in 2 days.
4. Circular: A->B->C->A within 4 hours.
5. High Risk Country: Receiver country in ['PK', 'AF', 'IR', 'KP', 'SY'] (simulate via branch code prefix 'HR_').
Return ONLY the python code.
"""
# print(call_llm(PROMPT)) # Uncomment to see LLM generated code

# ---------------------------------------------------------
# OPTION B: Robust Hardcoded Implementation (Faster/Deterministic for MVP)
# ---------------------------------------------------------

def generate_customers(n):
    custs = []
    for i in range(n):
        cust_id = f"CUST_{i:06d}"
        # 5% High Risk Customers
        risk = "HIGH" if random.random() < 0.05 else "NORMAL"
        custs.append({"cust_id": cust_id, "risk_rating": risk, "kyc_status": "COMPLETE"})
    return pd.DataFrame(custs)

def generate_accounts(customers):
    accs = []
    branches = [f"BR_{i:03d}" for i in range(100)] # 100 branches
    high_risk_branches = [f"HR_{i:02d}" for i in range(5)] # Simulated High Risk Jurisdiction branches
    all_branches = branches + high_risk_branches
    
    for _, row in customers.iterrows():
        # 1-3 accounts per customer
        for _ in range(random.randint(1, 3)):
            acc_id = f"ACC_{fake.unique.random_number(digits=10)}"
            branch = random.choice(all_branches)
            accs.append({"acc_id": acc_id, "cust_id": row.cust_id, "branch": branch, "open_date": fake.date_between(start_date='-5y', end_date='-1y')})
    return pd.DataFrame(accs)

def inject_patterns(df, accounts, customers):
    """Modifies dataframe to inject known ground truth patterns."""
    print("Injecting AML Patterns...")
    suspicious_indices = set()
    
    # Helper to pick random accounts
    def get_acc(cust_type='NORMAL'):
        cid = customers[customers.risk_rating==cust_type].sample(1).cust_id.values[0]
        return accounts[accounts.cust_id==cid].sample(1).acc_id.values[0]

    # 1. STRUCTURING: < 10L (1,000,000), multiple times, same sender
    for _ in range(50): # 50 structuring rings
        sender = get_acc()
        amt = round(random.uniform(9_00_000, 9_99_999), 2) # Just below 10L
        for _ in range(random.randint(3, 8)): # 3-8 deposits
            idx = len(df)
            df.loc[idx] = [f"TXN_{idx}", datetime.now() - timedelta(minutes=random.randint(1, 1440)), 
                           sender, get_acc(), "", "", amt, "NEFT", "", "", 1, "structuring"]
            suspicious_indices.add(idx)

    # 2. LAYERING: A->B->C->D (3 hops) fast
    for _ in range(30):
        chain = [get_acc() for _ in range(4)]
        base_amt = round(random.uniform(20_00_000, 50_00_000), 2)
        t = datetime.now() - timedelta(hours=random.randint(1, 720))
        for i in range(3):
            idx = len(df)
            df.loc[idx] = [f"TXN_{idx}", t + timedelta(minutes=random.randint(1, 15)), 
                           chain[i], chain[i+1], "", "", base_amt + random.uniform(-1000, 1000), "RTGS", "", "", 1, "layering"]
            suspicious_indices.add(idx)

    # 3. DORMANT WAKEUP
    for _ in range(20):
        acc = get_acc()
        # Ensure history exists but old
        wake_time = datetime.now() - timedelta(days=random.randint(1, 5))
        for _ in range(random.randint(3, 5)):
            idx = len(df)
            df.loc[idx] = [f"TXN_{idx}", wake_time + timedelta(hours=random.randint(1, 48)), 
                           acc, get_acc(), "", "", round(random.uniform(10_00_000, 1_00_00_000), 2), "IMPS", "", "", 1, "dormant_wakeup"]
            suspicious_indices.add(idx)

    # 4. CIRCULAR: A->B->C->A
    for _ in range(15):
        a, b, c = [get_acc() for _ in range(3)]
        amt = round(random.uniform(5_00_000, 20_00_000), 2)
        t = datetime.now() - timedelta(hours=random.randint(1, 480))
        for s, r in [(a,b), (b,c), (c,a)]:
            idx = len(df)
            df.loc[idx] = [f"TXN_{idx}", t + timedelta(minutes=random.randint(1, 60)), 
                           s, r, "", "", amt, "NEFT", "", "", 1, "circular"]
            suspicious_indices.add(idx)

    # 5. HIGH RISK COUNTRY (Branch prefix HR_)
    hr_branches = accounts[accounts.branch.str.startswith('HR_')].acc_id.tolist()
    if hr_branches:
        for _ in range(40):
            idx = len(df)
            df.loc[idx] = [f"TXN_{idx}", datetime.now() - timedelta(days=random.randint(1, 30)), 
                           get_acc(), random.choice(hr_branches), "", "", round(random.uniform(1_00_000, 10_00_000), 2), "SWIFT", "", "", 1, "high_risk_country"]
            suspicious_indices.add(idx)

    df.loc[list(suspicious_indices), 'is_suspicious'] = 1
    return df

def main():
    print("Generating Customers & Accounts...")
    customers = generate_customers(N_CUSTOMERS)
    accounts = generate_accounts(customers)
    acc_list = accounts.acc_id.tolist()
    branch_map = accounts.set_index('acc_id')['branch'].to_dict()
    cust_map = accounts.set_index('acc_id')['cust_id'].to_dict()

    print(f"Generating {N_TXNS} Normal Transactions...")
    # Vectorized normal generation
    txn_ids = [f"TXN_{i}" for i in range(N_TXNS)]
    timestamps = [datetime.now() - timedelta(days=random.randint(0, 180), hours=random.randint(0,23), minutes=random.randint(0,59)) for _ in range(N_TXNS)]
    senders = random.choices(acc_list, k=N_TXNS)
    receivers = random.choices(acc_list, k=N_TXNS)
    # Prevent self loops mostly
    receivers = [r if r!=s else random.choice(acc_list) for s,r in zip(senders, receivers)]
    amounts = [round(random.lognormvariate(8, 1.5), 2) for _ in range(N_TXNS)] # Log normal for realistic amounts
    txn_types = random.choices(['NEFT', 'RTGS', 'IMPS', 'UPI', 'CASH'], weights=[0.3, 0.1, 0.4, 0.15, 0.05], k=N_TXNS)
    
    df = pd.DataFrame({
        'txn_id': txn_ids, 'timestamp': timestamps, 'sender_acc': senders, 'receiver_acc': receivers,
        'sender_cust_id': [cust_map[s] for s in senders], 'receiver_cust_id': [cust_map[r] for r in receivers],
        'amount': amounts, 'txn_type': txn_types,
        'sender_branch': [branch_map[s] for s in senders], 'receiver_branch': [branch_map[r] for r in receivers],
        'is_suspicious': 0, 'suspicious_pattern': None
    })

    # Inject Patterns
    df = inject_patterns(df, accounts, customers)
    
    # Shuffle & Save
    df = df.sample(frac=1).reset_index(drop=True)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(df)} transactions to {OUTPUT_PATH}")
    print(f"   Suspicious: {df.is_suspicious.sum()} ({df.is_suspicious.mean()*100:.2f}%)")

if __name__ == "__main__":
    main()

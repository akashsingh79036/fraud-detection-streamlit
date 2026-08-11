import os
import sys
import pandas as pd

# Setup path to import core config tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import call_llm

print("--- Step 1: Reading Final Operational Alert File ---")
data_path = "data/operational_alerts_final.csv"

if not os.path.exists(data_path):
    print(f"❌ Error: {data_path} not found. Run core/scoring.py first!")
    sys.exit()

df = pd.read_csv(data_path)

# Isolate the exact #1 highest priority transaction on the list
top_target = df.iloc[0]

print(f"Target found: {top_target['txn_id']} | Priority Score: {top_target['investigation_priority']:.2f}")

print("\n--- Step 2: Requesting Automated Investigation Brief from NVIDIA NIM ---")
prompt = f"""
We need a standard banking compliance brief for our #1 highest-priority alert.
Here are the transaction details:
- Transaction ID: {top_target['txn_id']}
- Target Timestamp: {top_target['timestamp']}
- Account Sender ID: {top_target['sender_acc']}
- Account Receiver ID: {top_target['receiver_acc']}
- Transaction Amount: ${top_target['amount']:,.2f}
- Channel Type: {top_target['txn_type']}
- Behavior Rule Score: {top_target['rule_score']}
- AI Anomaly Confidence Score: {top_target['anomaly_score']:.4f}
- Injected Ground Truth Pattern: {top_target['suspicious_pattern']}

Please write a structured, 3-paragraph Case Investigation Memo. 
Include:
1. Executive Case Summary detailing the customer transaction behavior anomaly.
2. Technical Machine Learning Audit confirming why this specific row triggered high system metrics.
3. Operational Action Steps for filing a Suspicious Activity Report (SAR).
"""

# Call the API using your specialized Lead Investigator Persona
brief_content = call_llm(prompt=prompt, system_prompt="You are a Lead AML Forensic Investigator.")

print("\n=== AI FORENSIC INVESTIGATION MEMO ===")
print(brief_content)

# Save the final text record out to your project folder
brief_path = "artifacts/top_priority_investigation_brief.txt"
with open(brief_path, "w", encoding="utf-8") as f:
    f.write(brief_content)
print(f"\n💾 Document successfully saved to disc at: {brief_path}")

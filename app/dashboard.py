# app/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.engine import compute_behavioral_features, apply_rule_engine
from core.model import train_model, score_model, MODEL_PATH
from core.scoring import calculate_final_risk
from core.graph import build_suspicious_subgraph
import joblib

st.set_page_config(page_title="PSU Bank AML MVP", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data(path="data/raw/transactions.csv"):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df

@st.cache_resource
def run_pipeline(_df):
    # 1. Features
    df = compute_behavioral_features(_df)
    # 2. Rules
    df = apply_rule_engine(df)
    # 3. ML
    try:
        pipe = joblib.load(MODEL_PATH)
    except:
        pipe = train_model(df)
    df = score_model(df, pipe)
    # 4. Scoring
    df = calculate_final_risk(df)
    return df

# --- UI ---
st.title("🛡️ PSU Bank AML Detection MVP")
st.caption("Hybrid Rule-Based + AI Anomaly Detection | RBI AML Compliance")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    data_path = st.text_input("Data Path", "data/raw/transactions.csv")
    reload = st.button("🔄 Reload & Re-run Pipeline")
    st.markdown("---")
    st.info("**MVP Scope:**\n- Structuring & Spike Rules\n- Isolation Forest Anomaly\n- Risk Scoring (H/M/L)\n- Network Graph")

if 'df' not in st.session_state or reload:
    with st.spinner("Loading Data & Running Pipeline..."):
        raw_df = load_data(data_path)
        st.session_state.df = run_pipeline(raw_df)
    st.success("Pipeline Complete!")

df = st.session_state.df

# --- KPI Row ---
col1, col2, col3, col4 = st.columns(4)
total_txns = len(df)
susp_txns = df['is_suspicious'].sum() # Ground truth
high_risk = (df['risk_level'] == 'HIGH').sum()
flagged = df['ml_flag'].sum()

col1.metric("Total Transactions", f"{total_txns:,}")
col2.metric("Ground Truth Suspicious", f"{susp_txns:,}", help="Injected in synthetic data")
col3.metric("🚨 HIGH Risk Cases", f"{high_risk:,}", delta=f"{high_risk/total_txns*100:.2f}%")
col4.metric("🤖 ML Anomalies Flagged", f"{flagged:,}")

st.markdown("---")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Risk Overview", "🔍 Case Investigation", "🕸️ Network Graph", "📈 Model Performance"])

with tab1:
    st.subheader("Risk Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df, names='risk_level', title="Risk Level Distribution", color='risk_level',
                     color_discrete_map={'HIGH':'red', 'MEDIUM':'orange', 'LOW':'green'})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x='risk_score', color='risk_level', nbins=50, title="Risk Score Density")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top Triggered Rules")
    # Explode rules
    rules_exploded = df.explode('triggered_rules').dropna(subset=['triggered_rules'])
    rule_counts = rules_exploded['triggered_rules'].value_counts().reset_index()
    rule_counts.columns = ['Rule', 'Count']
    fig = px.bar(rule_counts, x='Rule', y='Count', title="Rule Trigger Frequency", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Prioritized Investigation Queue")
    st.caption("Sorted by Hybrid Priority (Risk Level + Rule Severity)")
    
    # Filters
    f_col1, f_col2 = st.columns(2)
    risk_filter = f_col1.multiselect("Risk Level", ['HIGH', 'MEDIUM', 'LOW'], default=['HIGH', 'MEDIUM'])
    pattern_filter = f_col2.multiselect("Pattern", df['suspicious_pattern'].dropna().unique().tolist())
    
    view_df = df[df['risk_level'].isin(risk_filter)]
    if pattern_filter:
        view_df = view_df[view_df['suspicious_pattern'].isin(pattern_filter)]
    
    # Display Columns
    disp_cols = ['txn_id', 'timestamp', 'sender_acc', 'receiver_acc', 'amount', 'txn_type', 
                 'risk_level', 'risk_score', 'rule_score', 'anomaly_score', 'triggered_rules', 'suspicious_pattern']
    
    st.dataframe(
        view_df[disp_cols].head(200),
        use_container_width=True,
        height=600,
        column_config={
            "amount": st.column_config.NumberColumn(format="₹%,.0f"),
            "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.2f"),
            "anomaly_score": st.column_config.ProgressColumn("Anomaly", min_value=0, max_value=1, format="%.2f"),
            "triggered_rules": st.column_config.ListColumn("Rules"),
        }
    )
    
    # Download
    csv = view_df[disp_cols].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Queue (CSV)", csv, "investigation_queue.csv", "text/csv")

with tab3:
    st.subheader("Suspicious Network Topology")
    st.caption("Top 50 Priority Cases + 1-Hop Context. Red=High Risk Nodes/Edges.")
    
    with st.spinner("Building Graph..."):
        net = build_suspicious_subgraph(df, top_n_cases=50)
        html_path = "temp_graph.html"
        net.save_graph(html_path)
        
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        st.components.v1.html(html, height=750, scrolling=True)

with tab4:
    st.subheader("Detection Performance (vs Synthetic Ground Truth)")
    # Only works because we have 'is_suspicious' column
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
    
    y_true = df['is_suspicious']
    y_score = df['risk_score'] # Use hybrid score
    
    # Threshold at HIGH risk
    y_pred = (df['risk_level'] == 'HIGH').astype(int)
    
    c1, c2 = st.columns(2)
    with c1:
        st.text("Classification Report (HIGH vs Rest)")
        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    
    with c2:
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Pred LOW/MED', 'Pred HIGH'], y=['Actual Normal', 'Actual Susp'],
                        title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig = px.area(x=recall, y=precision, title=f"Precision-Recall Curve (AUC={roc_auc_score(y_true, y_score):.3f})",
                  labels=dict(x="Recall", y="Precision"))
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=1, y1=0)
    st.plotly_chart(fig, use_container_width=True)

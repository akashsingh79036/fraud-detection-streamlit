# app/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import traceback

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Core Modules
from core.engine import compute_behavioral_features, apply_rule_engine
from core.model import load_or_train, score_model
from core.scoring import calculate_final_risk
from core.graph import build_suspicious_subgraph

st.set_page_config(
    page_title="PSU Bank AML Detection MVP", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# DATA LOADING & PIPELINE
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading raw transaction data...")
def load_data(path: str) -> pd.DataFrame:
    """Loads CSV, parses dates, basic validation."""
    p = Path(path)
    if not p.exists():
        st.error(f"File not found: {p.resolve()}")
        return pd.DataFrame()
    
    df = pd.read_csv(p, parse_dates=['timestamp'], low_memory=False)
    # Ensure required cols exist
    required = ['txn_id', 'timestamp', 'sender_acc', 'receiver_acc', 'amount', 'txn_type']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return pd.DataFrame()
    return df

# app/dashboard.py (Inside run_pipeline function)

@st.cache_resource(show_spinner="Running Detection Pipeline...")
def run_pipeline(_df: pd.DataFrame) -> pd.DataFrame:
    if _df.empty: return _df
    df = _df.copy()
    
    try:
        with st.spinner("Step 1/4: Computing Behavioral Features..."):
            df = compute_behavioral_features(df)
        
        with st.spinner("Step 2/4: Applying Rule Engine..."):
            df = apply_rule_engine(df)
        
        with st.spinner("Step 3/4: Scoring ML Models..."):
            artifact = load_or_train(df)
            df = score_model(df, artifact)
        
        with st.spinner("Step 4/4: Calculating Final Risk..."):
            # 🔑 PASS ARTIFACT HERE
            df = calculate_final_risk(df, artifact)
            
    except Exception as e:
        st.error(f"Pipeline Failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()
        
    return df

# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
st.title("🛡️ PSU Bank AML Detection MVP")
st.caption("Hybrid Rule-Based + AI Anomaly Detection | RBI AML Compliance")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Controls")
    data_path = st.text_input("Data Path", "data/raw/transactions.csv")
    
    col_a, col_b = st.columns(2)
    if col_a.button("🔄 Reload Data & Rerun", type="primary", use_container_width=True):
        st.session_state.clear() # Clear cache and state
        st.rerun()
    
    if col_b.button("🗑️ Clear Model Cache", use_container_width=True):
        import shutil
        model_dir = ROOT_DIR / "artifacts"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        st.success("Model artifacts deleted. Retraining on next run.")
        st.rerun()

    st.markdown("---")
    st.info("**MVP Scope:**\n- Structuring & Velocity Rules\n- Isolation Forest (Novelty)\n- XGBoost (Supervised Labels)\n- Risk Scoring (H/M/L)\n- Network Graph Investigation")

# Initialize Session State
if 'df' not in st.session_state:
    raw_df = load_data(data_path)
    if not raw_df.empty:
        with st.spinner("Initializing Pipeline... (First run trains models)"):
            st.session_state.df = run_pipeline(raw_df)
    else:
        st.stop()

df = st.session_state.df

# ---------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------
# app/dashboard.py (Lines 116-123)
# ---------------------------------------------------------
# KPI ROW (CRASH-PROOF)
# ---------------------------------------------------------
total_txns = len(df)

# Safe metric calculations with column checking fallback loops
susp_txns = int(df['is_suspicious'].sum()) if 'is_suspicious' in df.columns else 0
high_risk = int((df['risk_level'] == 'HIGH').sum()) if 'risk_level' in df.columns else 0
ml_flagged = int(df['ml_flag'].sum()) if 'ml_flag' in df.columns else 0
rule_flagged = int(df['rule_flag'].sum()) if 'rule_flag' in df.columns else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Transactions", f"{total_txns:,}")
col2.metric("Ground Truth Suspicious", f"{susp_txns:,}")
col3.metric("🚨 HIGH Risk Cases", f"{high_risk:,}")
col4.metric("🤖 ML Anomalies Flagged", f"{ml_flagged:,}")
col5.metric("📏 Rule Hits", f"{rule_flagged:,}")

st.markdown("---")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Risk Overview", 
    "🔍 Case Investigation", 
    "🕸️ Network Graph", 
    "📈 Model Performance",
    "🔮 Live Simulator"
])

# ==========================================================
# TAB 1: RISK OVERVIEW
# ==========================================================
with tab1:
    st.subheader("Risk Distribution")
    c1, c2 = st.columns(2)
    
    with c1:
        if 'risk_level' in df.columns:
            risk_counts = df['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            fig = px.pie(risk_counts, names='Risk Level', values='Count', 
                         title="Risk Level Distribution", hole=0.4,
                         color='Risk Level',
                         color_discrete_map={'HIGH':'#ff4444', 'MEDIUM':'#ffaa00', 'LOW':'#44ff44'})
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        if 'risk_score' in df.columns:
            fig = px.histogram(df, x='risk_score', color='risk_level', nbins=50, 
                               title="Risk Score Density",
                               color_discrete_map={'HIGH':'#ff4444', 'MEDIUM':'#ffaa00', 'LOW':'#44ff44'},
                               marginal="box")
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top Triggered Rules")
    if 'triggered_rules' in df.columns:
        # Explode list column
        rules_exploded = df.explode('triggered_rules').dropna(subset=['triggered_rules'])
        if not rules_exploded.empty:
            rule_counts = rules_exploded['triggered_rules'].value_counts().reset_index()
            rule_counts.columns = ['Rule', 'Count']
            fig = px.bar(rule_counts, x='Rule', y='Count', title="Rule Trigger Frequency", 
                         text_auto=True, color='Count', color_continuous_scale='Reds')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rules triggered in current dataset.")

# ==========================================================
# TAB 2: CASE INVESTIGATION
# ==========================================================
with tab2:
    st.subheader("Prioritized Investigation Queue")
    st.caption("Sorted by **Investigation Priority** (Risk Level × Rule Severity × Anomaly Score). Top 500 shown.")
    
    # Filters
    f_col1, f_col2, f_col3 = st.columns(3)
    risk_opts = ['HIGH', 'MEDIUM', 'LOW']
    risk_filter = f_col1.multiselect("Risk Level", risk_opts, default=['HIGH', 'MEDIUM'])
    
    pattern_opts = df['suspicious_pattern'].dropna().unique().tolist() if 'suspicious_pattern' in df.columns else []
    pattern_filter = f_col2.multiselect("Suspicious Pattern", pattern_opts, default=[])
    
    # ML Flag Filter
    ml_filter = f_col3.selectbox("ML Flag", ["All", "Flagged Only (1)", "Not Flagged (0)"])

    # Apply Filters
    view_df = df[df['risk_level'].isin(risk_filter)].copy()
    
    if pattern_filter:
        view_df = view_df[view_df['suspicious_pattern'].isin(pattern_filter)]
    
    if ml_filter == "Flagged Only (1)" and 'ml_flag' in view_df.columns:
        view_df = view_df[view_df['ml_flag'] == 1]
    elif ml_filter == "Not Flagged (0)" and 'ml_flag' in view_df.columns:
        view_df = view_df[view_df['ml_flag'] == 0]

    # Sort by Priority
    if 'investigation_priority' in view_df.columns:
        view_df = view_df.sort_values('investigation_priority', ascending=False)

    # Display Columns
    disp_cols = [
        'txn_id', 'timestamp', 'sender_acc', 'receiver_acc', 'amount', 'txn_type',
        'risk_level', 'risk_score', 'rule_score', 'anomaly_score', 
        'ml_flag', 'triggered_rules', 'suspicious_pattern'
    ]
    # Filter cols that actually exist
    disp_cols = [c for c in disp_cols if c in view_df.columns]

    st.dataframe(
        view_df[disp_cols].head(500),
        use_container_width=True,
        height=650,
        hide_index=True,
        column_config={
            "amount": st.column_config.NumberColumn("Amount (₹)", format="₹%,.0f"),
            "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=1, format="%.3f"),
            "anomaly_score": st.column_config.ProgressColumn("Anomaly Score", min_value=0, max_value=1, format="%.3f"),
            "rule_score": st.column_config.NumberColumn("Rule Score", format="%.1f"),
            "ml_flag": st.column_config.CheckboxColumn("ML Flag"),
            "triggered_rules": st.column_config.ListColumn("Triggered Rules", width="medium"),
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="DD-MMM-YYYY HH:mm"),
        }
    )
    
    # Download Button
    if not view_df.empty:
        csv = view_df[disp_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Filtered Queue (CSV)", 
            csv, 
            "investigation_queue.csv", 
            "text/csv",
            use_container_width=True
        )

# ==========================================================
# TAB 3: NETWORK GRAPH
# ==========================================================
with tab3:
    st.subheader("Suspicious Network Topology")
    st.caption("Top 50 Priority Cases + 1-Hop Context. **Red** = High Risk Nodes/Edges. Physics enabled.")
    
    if st.button("🔄 Rebuild Graph", key="rebuild_graph"):
        st.cache_data.clear() # Clear graph cache if any
        
    with st.spinner("Building Network Graph (Top 50 Priority Cases)..."):
        try:
            # Limit to top 50 priority for performance
            net = build_suspicious_subgraph(df, top_n_cases=50)
            html_path = ROOT_DIR / "temp_graph.html"
            net.save_graph(str(html_path))
            
            with open(html_path, 'r', encoding='utf-8') as f:
                html = f.read()
            
            st.components.v1.html(html, height=750, scrolling=True)
        except Exception as e:
            st.error(f"Graph Generation Failed: {e}")
            st.code(traceback.format_exc())

# ==========================================================
# TAB 4: MODEL PERFORMANCE
# ==========================================================
with tab4:
    st.subheader("Detection Performance (vs Synthetic Ground Truth)")
    st.caption("Metrics calculated on **Full Dataset** using `is_suspicious` column (Injected Labels).")
    
    if 'is_suspicious' not in df.columns:
        st.warning("Ground truth column `is_suspicious` not found. Cannot compute performance.")
    else:
        from sklearn.metrics import (classification_report, confusion_matrix, 
                                     roc_auc_score, average_precision_score, 
                                     precision_recall_curve, roc_curve)
        
        y_true = df['is_suspicious'].astype(int)
        
        # Use Hybrid Risk Score for curves (continuous)
        y_score = df['risk_score'] if 'risk_score' in df.columns else df.get('anomaly_score', 0)
        
        # Binary Prediction: HIGH Risk = 1
        y_pred = (df['risk_level'] == 'HIGH').astype(int)
        
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("**Classification Report (HIGH vs Rest)**")
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:.0f}"}))
            
            # Key Metrics
            auc_roc = roc_auc_score(y_true, y_score)
            auc_pr = average_precision_score(y_true, y_score)
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("ROC-AUC", f"{auc_roc:.4f}")
            m_col2.metric("PR-AUC (Avg Precision)", f"{auc_pr:.4f}", help="Primary metric for imbalanced data")

        with c2:
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig = px.imshow(cm, text_auto=True, 
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=['Pred LOW/MED', 'Pred HIGH'], 
                            y=['Actual Normal', 'Actual Suspicious'],
                            title="Confusion Matrix (HIGH Risk Threshold)",
                            color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        # Precision-Recall Curve
        st.markdown("**Precision-Recall Curve**")
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        
        fig_pr = px.area(
            x=recall, y=precision, 
            title=f"PR Curve (AUC={auc_pr:.3f}) | Baseline (Random) = {y_true.mean():.3f}",
            labels=dict(x="Recall", y="Precision"),
            template="plotly_white"
        )
        fig_pr.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=y_true.mean(), y1=y_true.mean())
        fig_pr.add_annotation(x=0.5, y=y_true.mean()+0.02, text="Random Baseline", showarrow=False, font_color="gray")
        st.plotly_chart(fig_pr, use_container_width=True)
        
        # ROC Curve
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC={auc_roc:.3f})", labels=dict(x="False Positive Rate", y="True Positive Rate"))
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc, use_container_width=True)
        
# =========================================================
# 🚀 TAB 5: LIVE PRODUCTION SIMULATOR (Paste at absolute bottom)
# =========================================================
with tab5:
    st.subheader("Playground: Inject a Real-Time Production Transaction")
    st.caption("Manually adjust telemetry parameters to see how the hybrid AI models respond instantly.")
    
    # Initialize separate state tracking variables for the live playground form
    if 'sim_result' not in st.session_state:
        st.session_state.sim_result = None
        st.session_state.sim_data = None

    # Organize input components cleanly using grid columns
    form_col1, form_col2, form_col3 = st.columns(3)
    
    with form_col1:
        sim_id = st.text_input("Transaction ID Reference", "TXN_HACKATHON_LIVE_01")
        sim_sender = st.text_input("Sender Account", "ACC_LIVE_TEST_77")
        sim_receiver = st.text_input("Receiver Account", "ACC_LIVE_TEST_88")
        
    with form_col2:
        sim_amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=5000000.0, value=945000.0, step=5000.0)
        sim_type = st.selectbox("Payment Gateway Channel", ["NEFT", "RTGS", "IMPS", "CASH"])
        
    with form_col3:
        sim_sender_br = st.text_input("Sender Branch Code", "BR_011")
        sim_receiver_br = st.selectbox("Receiver Branch Code (Risk Region)", ["BR_022", "BR_056", "HR_04 (High Risk Location)"])

    # Trigger action button
    if st.button("🚀 Process Live Entry through ML Pipeline", type="primary", use_container_width=True):
        with st.spinner("Executing real-time feature transformation and scoring mechanics..."):
            import numpy as np
            import time
            
            clean_rec_branch = "HR_04" if "HR_04" in sim_receiver_br else sim_receiver_br
            
            # 1. Assemble raw dictionary payload 
            mock_payload = {
                'txn_id': sim_id,
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sender_acc': sim_sender,
                'receiver_acc': sim_receiver,
                'sender_cust_id': 'CUST_MOCK_LIVE',
                'receiver_cust_id': 'CUST_MOCK_LIVE',
                'amount': float(sim_amount),
                'txn_type': sim_type,
                'sender_branch': sim_sender_br,
                'receiver_branch': clean_rec_branch,
                'is_suspicious': 0,
                'suspicious_pattern': np.nan
            }
            
            # 2. Append directly to our current session history array to calculate 24h rolling velocity
            extended_df = pd.concat([df, pd.DataFrame([mock_payload])], ignore_index=True)
            
            # 3. Process calculations on the fly using your core modules
            from core.engine import compute_behavioral_features, apply_rule_engine
            from core.model import load_or_train, score_model
            
            proc_df = compute_behavioral_features(extended_df)
            rule_df = apply_rule_engine(proc_df)
            
            artifact = load_or_train(rule_df)
            ml_scored_df = score_model(rule_df, artifact)
            final_scored_df = calculate_final_risk(ml_scored_df, artifact)
            
            # 4. Save results to Session State to keep them active across button clicks
            st.session_state.sim_result = final_scored_df[final_scored_df['txn_id'] == sim_id].iloc[0].to_dict()
            st.session_state.sim_data = mock_payload
            time.sleep(0.4)

    # =========================================================
    # RENDER ENGINE RESULTS (Decoupled from button click state)
    # =========================================================
    if st.session_state.sim_result is not None:
        live_result = st.session_state.sim_result
        
        st.markdown("---")
        st.markdown("### 📊 Pipeline Real-Time Risk Diagnostics")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        badge_color = "🔴" if live_result['risk_level'] == 'HIGH' else "🟡" if live_result['risk_level'] == 'MEDIUM' else "🟢"
        
        metric_col1.metric("Risk Status Level", f"{badge_color} {live_result['risk_level']}")
        metric_col2.metric("AI Score Weight", f"{live_result['anomaly_score']:.4f}")
        metric_col3.metric("Rule Score Metric", f"{live_result['rule_score']:.1f}")
        metric_col4.metric("Priority Rank Index", f"{live_result['investigation_priority']:.2f}")
        
        if live_result['risk_level'] == 'HIGH':
            st.error(f"⚠️ **Alert Triggered!** This transaction exhibits significant money laundering traits. Priority indexing score is **{live_result['investigation_priority']:.2f}**.")
            
            if sim_amount > 900000 and sim_amount < 1000000:
                st.info("💡 **Pipeline Insight:** Caught a potential **Structuring (Smurfing)** threat. The amount is deliberately engineered just under standard regulatory reporting limits ($1,000,000).")
            if "HR_" in str(live_result['receiver_branch']):
                st.info("💡 **Pipeline Insight:** Flagged a **Geographic Anomaly**. Funds are tracking directly into high-risk settlement jurisdictions.")
                
            # =========================================================
            # 🔮 INTEGRATED NVIDIA AI CO-PILOT (Safe State Layout)
            # =========================================================
            st.markdown("---")
            st.subheader("🤖 NVIDIA AI Co-Pilot Investigator")
            st.caption("Generate an instant compliance investigation report for this alert using Llama 3.1.")
            
            from config import call_llm
            
            if st.button("📝 Draft Official Case Memo", use_container_width=True):
                with st.spinner("NVIDIA NIM generating regulatory brief..."):
                    live_prompt = f"""
                    Write a brief 2-paragraph banking compliance memo for this live alert:
                    - Transaction ID: {live_result['txn_id']}
                    - Amount: ${live_result['amount']:,.2f}
                    - Channel: {live_result['txn_type']}
                    - Sender/Receiver: {live_result['sender_acc']} -> {live_result['receiver_acc']}
                    - AI Score: {live_result['anomaly_score']:.4f}
                    - System Priority Index: {live_result['investigation_priority']:.2f}
                    
                    Include a concise Case Summary and immediate next-step actions for the auditing team.
                    """
                    try:
                        ai_memo = call_llm(prompt=live_prompt, system_prompt="You are a Lead AML Forensic Investigator.")
                        st.info("📊 **Drafted Case Report:**")
                        st.write(ai_memo)
                    except Exception as ai_err:
                        st.error(f"Could not connect to NVIDIA NIM: {ai_err}")
        else:
            st.success("✅ **Clear Status:** No critical behavioral anomalies detected. Transaction fits expected baseline parameter tracks.")

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
total_txns = len(df)
susp_txns = int(df['is_suspicious'].sum()) if 'is_suspicious' in df.columns else 0
high_risk = int((df['risk_level'] == 'HIGH').sum())
ml_flagged = int(df['ml_flag'].sum()) if 'ml_flag' in df.columns else 0
rule_flagged = int(df['rule_flag'].sum()) if 'rule_flag' in df.columns else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Transactions", f"{total_txns:,}")
col2.metric("Ground Truth Suspicious", f"{susp_txns:,}", help="Injected synthetic labels")
col3.metric("🚨 HIGH Risk Cases", f"{high_risk:,}", delta=f"{high_risk/total_txns*100:.2f}%", delta_color="inverse")
col4.metric("🤖 ML Anomalies Flagged", f"{ml_flagged:,}")
col5.metric("📏 Rule Hits", f"{rule_flagged:,}")

st.markdown("---")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Risk Overview", 
    "🔍 Case Investigation", 
    "🕸️ Network Graph", 
    "📈 Model Performance"
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

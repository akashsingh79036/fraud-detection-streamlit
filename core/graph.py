import networkx as nx
from pyvis.network import Network
import pandas as pd

def build_suspicious_subgraph(df: pd.DataFrame, top_n_cases: int = 50) -> Network:
    """Builds interactive HTML graph for top N suspicious transactions."""
    # Filter high priority
    sub_df = df.nlargest(top_n_cases, 'investigation_priority')
    
    # Get all accounts involved in these txns
    accounts = set(sub_df['sender_acc']).union(set(sub_df['receiver_acc']))
    
    # Pull ALL transactions between these accounts (1-hop context)
    context_df = df[df['sender_acc'].isin(accounts) & df['receiver_acc'].isin(accounts)]
    
    G = nx.DiGraph()
    
    # Add Nodes
    for acc in accounts:
        node_risk = sub_df[sub_df['sender_acc']==acc]['risk_score'].max()
        if pd.isna(node_risk): node_risk = 0
        G.add_node(acc, 
                   title=f"Account: {acc}\\nMax Risk: {node_risk:.2f}", 
                   color="#ff4444" if node_risk > 0.6 else "#ffaa00" if node_risk > 0.3 else "#44ff44",
                   size=10 + node_risk * 30)
    
    # Add Edges (Aggregate)
    edge_weights = context_df.groupby(['sender_acc', 'receiver_acc']).agg(
        total_amt=('amount', 'sum'),
        txn_count=('txn_id', 'count'),
        max_risk=('risk_score', 'max'),
        patterns=('suspicious_pattern', lambda x: ', '.join(x.dropna().unique()))
    ).reset_index()
    
    for _, row in edge_weights.iterrows():
        G.add_edge(row['sender_acc'], row['receiver_acc'],
                   title=f"Txns: {row['txn_count']}\\nTotal: {row['total_amt']:,.0f}\\nPatterns: {row['patterns']}",
                   value=row['txn_count'], # Edge thickness
                   color="#ff0000" if row['max_risk'] > 0.6 else "#888888")
    
    # PyVis
    net = Network(height="700px", width="100%", directed=True, notebook=False, cdn_resources="remote")
    net.from_nx(G)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100)
    net.show_buttons(filter_=['physics'])
    return net

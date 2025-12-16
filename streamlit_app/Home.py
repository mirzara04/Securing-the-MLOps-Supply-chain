# streamlit_app/Home.py
import streamlit as st
import mlflow
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Secure MLOps Dashboard", layout="wide")

st.title("🛡️ Secure MLOps: Fraud Detection Monitoring")

# Sidebar for MLflow Tracking URI
st.sidebar.header("Configuration")
mlflow_uri = st.sidebar.text_input("MLflow Tracking URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_uri)

st.write("""
### Project Overview
This dashboard visualizes the impact of **Data Poisoning** on model integrity and the effectiveness of **RONI** defenses.
""")

# Fetch latest runs
try:
    runs = mlflow.search_runs(experiment_names=["Fraud_Detection_Security"])
    if not runs.empty:
        st.subheader("Recent Training Runs")
        st.dataframe(runs[['run_id', 'params.poison_rate', 'params.defense_enabled', 'metrics.auc_score', 'start_time']])
        
        # Visualization
        fig = px.scatter(runs, x="params.poison_rate", y="metrics.auc_score", 
                         color="params.defense_enabled", title="Attack Impact: AUC vs Poison Rate")
        st.plotly_chart(fig)
    else:
        st.info("No runs found. Go to the Attack Lab to train your first model.")
except Exception as e:
    st.error(f"Could not connect to MLflow: {e}")
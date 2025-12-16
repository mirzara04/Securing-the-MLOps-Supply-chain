# streamlit_app/pages/Attack_Lab.py
import streamlit as st
import subprocess
import sys

st.title("🧪 Attack Simulation Laboratory")

st.write("Configure a poisoning attack and observe how the MLOps pipeline reacts.")

with st.form("attack_form"):
    poison_rate = st.slider("Poisoning Rate (Label Flipping)", 0.0, 0.5, 0.1)
    defense = st.checkbox("Enable RONI Defense")
    submit = st.form_submit_button("Launch Attack & Retrain")

if submit:
    st.warning(f"Starting training run: Poison={poison_rate}, Defense={defense}...")
    
    # Trigger the training script as a subprocess
    # In production, use a task queue like Celery
    cmd = [sys.executable, "src/train.py"] 
    # Note: Modify train.py to accept arguments if you want to pass these sliders
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        st.success("Training Complete! Check the Home page for updated metrics.")
    else:
        st.error(f"Training Failed: {result.stderr}")
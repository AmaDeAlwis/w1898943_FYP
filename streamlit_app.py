import streamlit as st
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import joblib
from torch_geometric.data import Data
from pymongo import MongoClient
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from gcn_model_class import SurvivalGNN

# Load saved StandardScaler (if saved during training)
scaler = joblib.load("scaler.pkl")  # Make sure you've saved this earlier

# Load GCN model
gcn_model = SurvivalGNN(
    in_channels=15,  # Adjust if needed
    hidden_channels=64,
    out_channels_time=1,
    out_channels_event=1
)
gcn_model.load_state_dict(torch.load(".streamlit/gcn_model.pt"))
gcn_model.eval()

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["breast_cancer_survival"]
collection = db["patient_records"]


# Set layout
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
h1 {
    text-align: center;
    color: #FFFFFF;
}

.section-title {
    font-size: 20px;
    font-weight: bold;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
    color: #ad1457;
}

/* Apply custom style to the Predict and Reset buttons */
button[aria-label="🔍 Predict"],
button[aria-label="🔄 Reset"] {
    background-color: #ad1457 !important;
    color: white !important;
    font-weight: bold !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    margin-top: 1rem !important;
    cursor: pointer !important;
}

/* Inputs */
input, select, textarea {
    border-radius: 10px !important;
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

# --- Handle Reset logic ---
if "reset" in st.query_params:
    st.query_params.clear()
    st.rerun()

# --- UI Start ---
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown("<h1> Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Fill in the details below to generate predictions and insights.</p>", unsafe_allow_html=True)

# --- FORM ---
with st.form("input_form", clear_on_submit=False):
    st.markdown("<div class='section-title'>🧬 Clinical Data</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=20, max_value=96)
        menopausal_status = st.selectbox("Menopausal Status", ["Pre-menopausal", "Post-menopausal"])
        tumor_stage = st.selectbox("Tumor Stage", [1, 2, 3, 4])
        lymph_nodes_examined = st.number_input("Lymph Nodes Examined", min_value=0, max_value=50)

    with col2:
        er_status = st.selectbox("ER Status", ["Positive", "Negative"])
        pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
        her2_status = st.selectbox("HER2 Status", ["Neutral", "Loss", "Gain", "Undef"])

    st.markdown("<div class='section-title'>💊 Treatment Data</div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        chemotherapy = st.selectbox("Chemotherapy", ["Yes", "No"])
        surgery = st.selectbox("Surgery Type", ["Breast-conserving", "Mastectomy"])
    with col4:
        radiotherapy = st.selectbox("Radiotherapy", ["Yes", "No"])
        hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])

    # Buttons inside the form
    colA, colB = st.columns(2)
    with colA:
        reset = st.form_submit_button("🔄 Reset")
    with colB:
        predict = st.form_submit_button("🔍 Predict")

# --- Button Logic ---
if reset:
    st.query_params["reset"] = "true"
    st.rerun()

if predict:
    st.success("Prediction functionality coming soon...")

# Close container
st.markdown("</div>", unsafe_allow_html=True)

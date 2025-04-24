import streamlit as st
import torch
import numpy as np
import joblib
from torch_geometric.data import Data
from pymongo import MongoClient
import datetime
from gcn_model_class import SurvivalGNN

# Set up the app
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# Load model and scaler
gcn_model = SurvivalGNN(in_channels=15, out_channels_time=1, out_channels_event=1)
gcn_model.load_state_dict(torch.load(".streamlit/gcn_model.pt", map_location=torch.device('cpu')))
gcn_model.eval()
scaler = joblib.load("scaler.pkl")

# MongoDB connection
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# Custom CSS
st.markdown("""
<style>
h1 { text-align: center; color: #FFFFFF; }
.section-title { font-size: 20px; font-weight: bold; margin-top: 2rem; margin-bottom: 0.5rem; color: #ad1457; }
button[kind="primary"] {
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
input, select, textarea { border-radius: 10px !important; cursor: pointer !important; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown("<h1> Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Fill in the details below to generate predictions and insights.</p>", unsafe_allow_html=True)

# --- Handle RESET from query param ---
if st.query_params.get("reset"):
    for key in [
        "age", "menopausal_status", "tumor_stage", "lymph_nodes_examined",
        "er_status", "pr_status", "her2_status", "chemotherapy",
        "surgery", "radiotherapy", "hormone_therapy"]:
        st.session_state.pop(key, None)
    st.query_params.clear()
    st.rerun()

# --- Form ---
with st.form("input_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=20, max_value=96, key="age") if "age" in st.session_state else st.number_input("Age", min_value=20, max_value=96, key="age")
        menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="menopausal_status")
        tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="tumor_stage")
        lymph_nodes_examined = st.number_input("Lymph Nodes Examined", min_value=0, max_value=50, key="lymph_nodes_examined") if "lymph_nodes_examined" in st.session_state else st.number_input("Lymph Nodes Examined", min_value=0, max_value=50, key="lymph_nodes_examined")
    with col2:
        er_status = st.selectbox("ER Status", ["", "Positive", "Negative"], key="er_status")
        pr_status = st.selectbox("PR Status", ["", "Positive", "Negative"], key="pr_status")
        her2_status = st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"], key="her2_status")

    col3, col4 = st.columns(2)
    with col3:
        chemotherapy = st.selectbox("Chemotherapy", ["", "Yes", "No"], key="chemotherapy")
        surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"], key="surgery")
    with col4:
        radiotherapy = st.selectbox("Radiotherapy", ["", "Yes", "No"], key="radiotherapy")
        hormone_therapy = st.selectbox("Hormone Therapy", ["", "Yes", "No"], key="hormone_therapy")

    colA, colB = st.columns(2)
    with colA:
        reset = st.form_submit_button("RESET")
    with colB:
        predict = st.form_submit_button("PREDICT")

if reset:
    st.query_params["reset"] = "true"
    st.rerun()

# PREDICT logic here (same as previous, omitted for brevity)

st.markdown("</div>", unsafe_allow_html=True)

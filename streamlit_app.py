import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import joblib
from torch_geometric.data import Data
from pymongo import MongoClient
import datetime
from gcn_model_class import SurvivalGNN

# Configuration
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# Load model
gcn_model = SurvivalGNN(in_channels=15, out_channels_time=1, out_channels_event=1)
gcn_model.load_state_dict(torch.load(".streamlit/gcn_model.pt", map_location=torch.device("cpu")))
gcn_model.eval()
scaler = joblib.load("scaler.pkl")

# MongoDB Connection
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# Initialize Patient ID safely
if "patient_id" not in st.session_state:
    st.session_state["patient_id"] = ""

# Field Keys
field_keys = [
    "age", "menopausal_status", "tumor_stage", "lymph_nodes_examined",
    "er_status", "pr_status", "her2_status", "chemotherapy",
    "surgery", "radiotherapy", "hormone_therapy"
]

# CSS Styling
st.markdown("""
<style>
h1 {
    color: #ad1457 !important;
    text-align: center;
    font-weight: bold;
}
.section-title {
    font-size: 20px;
    font-weight: bold;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
    color: #ad1457;
}
.stButton button {
    background-color: #ad1457 !important;
    color: white !important;
    font-weight: bold;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1> Breast Cancer Survival Prediction </h1>", unsafe_allow_html=True)

# Patient ID Section
st.markdown("<p class='section-title'>Patient Information</p>", unsafe_allow_html=True)
patient_id = st.text_input("Patient ID (Required)", value=st.session_state.get("patient_id", ""), key="patient_id")

# Clinical Information
st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.text_input("Age", value=st.session_state.get("age", ""), key="age")
    if st.session_state.get("age", ""):
        if not st.session_state.age.isdigit():
            st.warning("Age must be a number.")
        elif int(st.session_state.age) < 20:
            st.warning("Age must be at least 20.")

    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"],
                                     index=0 if "menopausal_status" not in st.session_state else
                                     ["", "Pre-menopausal", "Post-menopausal"].index(st.session_state["menopausal_status"]),
                                     key="menopausal_status")

    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4],
                               index=0 if "tumor_stage" not in st.session_state else
                               ["", 1, 2, 3, 4].index(st.session_state["tumor_stage"]),
                               key="tumor_stage")

    lymph_nodes_examined = st.text_input("Lymph Nodes Examined", value=st.session_state.get("lymph_nodes_examined", ""),
                                         key="lymph_nodes_examined")
    if st.session_state.get("lymph_nodes_examined", ""):
        if not st.session_state.lymph_nodes_examined.isdigit():
            st.warning("Lymph Nodes must be a number.")
        elif int(st.session_state.lymph_nodes_examined) < 0:
            st.warning("Lymph Nodes must be 0 or more.")

with col2:
    er_status = st.selectbox("ER Status", ["", "Positive", "Negative"],
                             index=0 if "er_status" not in st.session_state else
                             ["", "Positive", "Negative"].index(st.session_state["er_status"]),
                             key="er_status")

    pr_status = st.selectbox("PR Status", ["", "Positive", "Negative"],
                             index=0 if "pr_status" not in st.session_state else
                             ["", "Positive", "Negative"].index(st.session_state["pr_status"]),
                             key="pr_status")

    her2_status = st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"],
                               index=0 if "her2_status" not in st.session_state else
                               ["", "Neutral", "Loss", "Gain", "Undef"].index(st.session_state["her2_status"]),
                               key="her2_status")

# (The rest of the code continues exactly as you had previously, no change to your Prediction logic, Reset button etc.)

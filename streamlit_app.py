import streamlit as st
# Set layout
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

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
gcn_model = SurvivalGNN(
    in_channels=15,  # replace with actual number of features
    out_channels_time=1,
    out_channels_event=1
)
gcn_model.load_state_dict(torch.load(".streamlit/gcn_model.pt", map_location=torch.device('cpu')))
gcn_model.eval()

# Connect to MongoDB
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]




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
button[aria-label=" Predict"],
button[aria-label=" Reset"] {
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
    st.markdown("<div class='section-title'> Clinical Data</div>", unsafe_allow_html=True)
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

    st.markdown("<div class='section-title'> Treatment Data</div>", unsafe_allow_html=True)
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
        reset = st.form_submit_button("RESET")
    with colB:
        predict = st.form_submit_button("PREDICT")

# --- Button Logic ---
if reset:
    st.query_params["reset"] = "true"
    st.rerun()

if predict:
    # Encode categorical inputs
    menopausal_status = 1 if menopausal_status == "Post-menopausal" else 0
    er_status = 1 if er_status == "Positive" else 0
    pr_status = 1 if pr_status == "Positive" else 0
    # One-hot encode HER2 Status
    her2_neutral = 1 if her2_status == "Neutral" else 0
    her2_loss = 1 if her2_status == "Loss" else 0
    her2_gain = 1 if her2_status == "Gain" else 0
    her2_undef = 1 if her2_status == "Undef" else 0

    chemotherapy = 1 if chemotherapy == "Yes" else 0
    radiotherapy = 1 if radiotherapy == "Yes" else 0
    hormone_therapy = 1 if hormone_therapy == "Yes" else 0
    # One-hot encode Surgery Type (Breast-conserving vs Mastectomy)
    surgery_conserving = 1 if surgery == "Breast-conserving" else 0
    surgery_mastectomy = 1 if surgery == "Mastectomy" else 0

        # Input features (must match training feature order!)
    input_features = np.array([
        age,
        chemotherapy,
        er_status,
        hormone_therapy,
        menopausal_status,
        lymph_nodes_examined,
        pr_status,
        radiotherapy,
        tumor_stage,
        surgery_conserving,
        surgery_mastectomy,
        her2_gain,
        her2_loss,
        her2_neutral,
        her2_undef
    ]).reshape(1, -1)



    # Scale features
    input_scaled = scaler.transform(input_features)
    x_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Create dummy graph (single-node, self-loop)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    graph_data = Data(x=x_tensor, edge_index=edge_index)

    # Predict
    gcn_model.eval()
    with torch.no_grad():
        time_output, event_output = gcn_model(graph_data)
        survival_5yr = torch.sigmoid(time_output[0]).item()  # Assuming 5-year
        survival_10yr = torch.sigmoid(event_output[0]).item()  # Assuming 10-year
# Display Predictions Nicely
    st.markdown(f"""
    <div style='
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(220, 20, 60, 0.15);
        margin-top: 2rem;
        text-align: center;
        width: 90%;
        margin-left: auto;
        margin-right: auto;
    '>
        <h3 style='color: #c2185b;'>🎯 Survival Predictions</h3>

        <div style='margin-top: 1rem;'>
            <p style='font-size: 22px; font-weight: bold; color: #880e4f;'>
                🩺 5-Year Survival Probability:
                <span style="color:#d81b60;">{survival_5yr:.2f}</span>
            </p>
        </div>

        <div style='margin-top: 1.5rem;'>
            <p style='font-size: 22px; font-weight: bold; color: #880e4f;'>
                🩺 10-Year Survival Probability:
                <span style="color:#d81b60;">{survival_10yr:.2f}</span>
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)
  

# Save to MongoDB
    patient_data = {
        "timestamp": datetime.datetime.now(),
        "age": age,
        "menopausal_status": menopausal_status,
        "tumor_stage": tumor_stage,
        "lymph_nodes_examined": lymph_nodes_examined,
        "er_status": er_status,
        "pr_status": pr_status,
        "her2_status": her2_status,
        "chemotherapy": chemotherapy,
        "radiotherapy": radiotherapy,
        "hormone_therapy": hormone_therapy,
        "surgery": surgery,
        "survival_5yr": survival_5yr,
        "survival_10yr": survival_10yr
    }

    collection.insert_one(patient_data)
       
    st.markdown("""
         <div style='
            margin-top: 1.5rem;
            background-color: #fce4ec;
            padding: 1rem;
            border-radius: 15px;
            color: #880e4f;
            font-weight: bold;
            text-align: center;
        '>
             Patient prediction record successfully saved to MongoDB Atlas.
        </div>
        """, unsafe_allow_html=True)

# Close container
st.markdown("</div>", unsafe_allow_html=True)

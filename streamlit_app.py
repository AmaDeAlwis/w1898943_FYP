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

# Title
st.markdown("<h1> Breast Cancer Survival Prediction </h1>", unsafe_allow_html=True)

# --- Patient ID ---
st.markdown("<p class='section-title'>Patient Information</p>", unsafe_allow_html=True)
patient_id = st.text_input("Patient ID (Required)", key="patient_id")

# --- Clinical Information ---
st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age", key="age")
    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="menopausal_status")
    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="tumor_stage")
    lymph_nodes_examined = st.text_input("Lymph Nodes Examined", key="lymph_nodes_examined")
with col2:
    er_status = st.selectbox("ER Status", ["", "Positive", "Negative"], key="er_status")
    pr_status = st.selectbox("PR Status", ["", "Positive", "Negative"], key="pr_status")
    her2_status = st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"], key="her2_status")

# --- Treatment Information ---
st.markdown("<p class='section-title'>Treatment Information</p>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    chemotherapy = st.selectbox("Chemotherapy", ["", "Yes", "No"], key="chemotherapy")
    surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"], key="surgery")
with col4:
    radiotherapy = st.selectbox("Radiotherapy", ["", "Yes", "No"], key="radiotherapy")
    hormone_therapy = st.selectbox("Hormone Therapy", ["", "Yes", "No"], key="hormone_therapy")

# --- Buttons ---
left, right = st.columns(2)
with left:
    if st.button("RESET"):
        for k in field_keys + ["patient_id"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

with right:
    predict_clicked = st.button("PREDICT")

# --- Prediction Logic ---
if predict_clicked:
    if not patient_id:
        st.warning("Please enter a Patient ID")
    elif not all(st.session_state.get(k) for k in field_keys):
        st.warning("Please fill all fields")
    elif not age.isdigit() or int(age) < 20:
        st.warning("Age must be a number and at least 20")
    elif not lymph_nodes_examined.isdigit() or int(lymph_nodes_examined) < 0:
        st.warning("Lymph Nodes must be a non-negative number")
    else:
        # Preprocessing
        features = np.array([
            int(age),
            1 if chemotherapy == "Yes" else 0,
            1 if er_status == "Positive" else 0,
            1 if hormone_therapy == "Yes" else 0,
            1 if menopausal_status == "Post-menopausal" else 0,
            int(lymph_nodes_examined),
            1 if pr_status == "Positive" else 0,
            1 if radiotherapy == "Yes" else 0,
            int(tumor_stage),
            1 if surgery == "Breast-conserving" else 0,
            1 if surgery == "Mastectomy" else 0,
            1 if her2_status == "Gain" else 0,
            1 if her2_status == "Loss" else 0,
            1 if her2_status == "Neutral" else 0,
            1 if her2_status == "Undef" else 0,
        ]).reshape(1, -1)

        features_scaled = scaler.transform(features)
        graph_data = Data(x=torch.tensor(features_scaled, dtype=torch.float32), edge_index=torch.tensor([[0], [0]], dtype=torch.long))

        with torch.no_grad():
            time_out, event_out = gcn_model(graph_data)
            survival_5yr = torch.sigmoid(time_out[0]).item()
            survival_10yr = torch.sigmoid(event_out[0]).item()

        # Save to MongoDB
        patient_record = {
            "patient_id": patient_id,
            "timestamp": datetime.datetime.now(),
            "survival_5yr": survival_5yr,
            "survival_10yr": survival_10yr
        }
        collection.insert_one(patient_record)

        st.success("✅ Patient record successfully saved!")

        # --- Display Results ---
        st.markdown("<h2 style='color:#ad1457;'>Results Overview</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.bar(["5-Year", "10-Year"], [survival_5yr, survival_10yr], color="#FF69B4", width=0.5)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability", fontsize=8)
            for i, v in enumerate([survival_5yr, survival_10yr]):
                ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold', fontsize=8)
            st.pyplot(fig)

        with col2:
            if survival_5yr > 0.8:
                risk = ("High Survival Chance", "#d4edda")
                recommendation = "Patient shows a high probability. Continue standard monitoring."
            elif survival_5yr > 0.6:
                risk = ("Moderate Survival Chance", "#fff3cd")
                recommendation = "Patient shows moderate probability. Consider more frequent follow-up."
            else:
                risk = ("Low Survival Chance", "#f8d7da")
                recommendation = "Patient shows low probability. Consider aggressive treatment planning."

            st.markdown(f"""
                <div style='background-color:{risk[1]};padding:10px;border-radius:10px;margin-bottom:10px;'>
                <b>{risk[0]}</b>
                </div>
                <div style='background-color:#e3e4fa;padding:10px;border-radius:10px;'>
                {recommendation}
                </div>
            """, unsafe_allow_html=True)

        with col3:
            fig2, ax2 = plt.subplots(figsize=(2, 2))
            times = np.linspace(0, 1, 50)
            survival_curve = 1 - (1 - survival_5yr) * times
            ax2.plot(times, survival_curve, color="#FF69B4")
            ax2.set_ylim(0, 1)
            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Time", fontsize=8)
            ax2.set_ylabel("Survival Probability", fontsize=8)
            ax2.set_title("Survival Curve", fontsize=10)
            st.pyplot(fig2)

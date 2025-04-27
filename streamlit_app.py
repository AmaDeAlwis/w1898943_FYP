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
patient_id = st.text_input("Patient ID (Required)", value=st.session_state["patient_id"], key="patient_id")

# Previous Predictions
if patient_id:
    previous_records = list(collection.find({"patient_id": patient_id}))
    if previous_records:
        with st.expander(" View Previous Predictions for this Patient ID"):
            for record in previous_records:
                st.write(f"Date: {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"5-Year Survival: {record['survival_5yr']:.2f}")
                st.write(f"10-Year Survival: {record['survival_10yr']:.2f}")
                st.markdown("---")

# Clinical and Treatment Inputs (same as your version)
# ... [Skipping this part since you know it remains same]

# After PREDICTION ---

if predict_clicked:
    # [... After your prediction and survival_5yr, survival_10yr values are obtained]

    st.markdown("""
        <div style='background-color: #d4edda; padding: 1rem; border-radius: 10px;
                    color: #155724; border: 1px solid #c3e6cb;
                    margin-top: 1.5rem; font-weight: 500;'>
             ✅ Patient record successfully saved!
        </div>
    """, unsafe_allow_html=True)

    # Visualizations - All side by side now
    chart_col, tag_col, rec_col = st.columns([1, 1, 2])

    with chart_col:
        fig, ax = plt.subplots(figsize=(3, 2))
        bars = ax.bar(["5-Year", "10-Year"], [survival_5yr, survival_10yr], color="#FF69B4", width=0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Survival Probability", fontsize=10)
        ax.set_title("Survival at 5 and 10 Years", fontsize=12, fontweight="bold", pad=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, value in zip(bars, [survival_5yr, survival_10yr]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        st.pyplot(fig)

    with tag_col:
        if survival_5yr > 0.80:
            st.markdown("""
                <div style='background-color: #d4edda; padding: 1rem; border-radius: 10px;
                            color: #155724; text-align:center; font-weight: bold;'>
                    High Survival Chance
                </div>
            """, unsafe_allow_html=True)
        elif 0.60 < survival_5yr <= 0.80:
            st.markdown("""
                <div style='background-color: #fff3cd; padding: 1rem; border-radius: 10px;
                            color: #856404; text-align:center; font-weight: bold;'>
                    Moderate Survival Chance
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background-color: #f8d7da; padding: 1rem; border-radius: 10px;
                            color: #721c24; text-align:center; font-weight: bold;'>
                    Low Survival Chance
                </div>
            """, unsafe_allow_html=True)

    with rec_col:
        if survival_5yr > 0.80:
            st.success("Patient shows a high probability of 5-year survival. Continue standard monitoring.")
        elif 0.60 < survival_5yr <= 0.80:
            st.warning("Patient shows moderate probability. Consider more frequent follow-up.")
        else:
            st.error("Patient shows low probability. Consider aggressive treatment planning.")

# End of code.

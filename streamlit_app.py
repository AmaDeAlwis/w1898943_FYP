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
.container-box {
    background-color: #fff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1> Breast Cancer Survival Prediction </h1>", unsafe_allow_html=True)

# Patient ID Section
st.markdown("<p class='section-title'>Patient Information</p>", unsafe_allow_html=True)
patient_id = st.text_input("Patient ID (Required)", value=st.session_state["patient_id"], key="patient_id")

# Clinical Info
st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.session_state.age = st.text_input("Age", value=st.session_state.get("age", ""))
    if st.session_state.age.strip() and (not st.session_state.age.isdigit() or int(st.session_state.age) < 20):
        st.warning("Age must be a number and at least 20.")
    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="menopausal_status")
    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="tumor_stage")
    lymph_nodes_examined = st.text_input("Lymph Nodes Examined", value=st.session_state.get("lymph_nodes_examined", ""), key="lymph_nodes_examined")
with col2:
    er_status = st.selectbox("ER Status", ["", "Positive", "Negative"], key="er_status")
    pr_status = st.selectbox("PR Status", ["", "Positive", "Negative"], key="pr_status")
    her2_status = st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"], key="her2_status")

# Treatment Info
st.markdown("<p class='section-title'>Treatment Information</p>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    chemotherapy = st.selectbox("Chemotherapy", ["", "Yes", "No"], key="chemotherapy")
    surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"], key="surgery")
with col4:
    radiotherapy = st.selectbox("Radiotherapy", ["", "Yes", "No"], key="radiotherapy")
    hormone_therapy = st.selectbox("Hormone Therapy", ["", "Yes", "No"], key="hormone_therapy")

# Buttons
left, right = st.columns(2)
with left:
    if st.button("RESET"):
        for k in list(st.session_state.keys()):
            if k in field_keys + ["patient_id"]:
                del st.session_state[k]
        st.rerun()
with right:
    predict_clicked = st.button("PREDICT")

if predict_clicked:
    if not st.session_state.patient_id:
        st.warning("Please enter a Patient ID.")
    else:
        required_fields = [st.session_state.get(k, "") for k in field_keys]
        if "" in required_fields:
            st.warning("Please complete all fields!")
        else:
            # Preprocess inputs
            age = int(st.session_state.age)
            lymph_nodes_examined = int(st.session_state.lymph_nodes_examined)
            menopausal_status = 1 if st.session_state.menopausal_status == "Post-menopausal" else 0
            er_status = 1 if st.session_state.er_status == "Positive" else 0
            pr_status = 1 if st.session_state.pr_status == "Positive" else 0
            her2_map = {"Neutral": [1,0,0,0], "Loss": [0,1,0,0], "Gain": [0,0,1,0], "Undef": [0,0,0,1]}
            her2_neutral, her2_loss, her2_gain, her2_undef = her2_map.get(st.session_state.her2_status, [0,0,0,0])
            chemotherapy = 1 if st.session_state.chemotherapy == "Yes" else 0
            radiotherapy = 1 if st.session_state.radiotherapy == "Yes" else 0
            hormone_therapy = 1 if st.session_state.hormone_therapy == "Yes" else 0
            surgery_conserving = 1 if st.session_state.surgery == "Breast-conserving" else 0
            surgery_mastectomy = 1 if st.session_state.surgery == "Mastectomy" else 0
            tumor_stage = int(st.session_state.tumor_stage)

            features = np.array([
                age, chemotherapy, er_status, hormone_therapy, menopausal_status,
                lymph_nodes_examined, pr_status, radiotherapy, tumor_stage,
                surgery_conserving, surgery_mastectomy, her2_gain,
                her2_loss, her2_neutral, her2_undef
            ]).reshape(1, -1)

            scaled = scaler.transform(features)
            data_graph = Data(x=torch.tensor(scaled, dtype=torch.float32), edge_index=torch.tensor([[0],[0]], dtype=torch.long))

            with torch.no_grad():
                out_time, out_event = gcn_model(data_graph)
                survival_5yr = torch.sigmoid(out_time[0]).item()
                survival_10yr = torch.sigmoid(out_event[0]).item()

            # Insert Record
            collection.insert_one({
                "patient_id": st.session_state.patient_id,
                "timestamp": datetime.datetime.now(),
                "survival_5yr": survival_5yr,
                "survival_10yr": survival_10yr
            })

            # Display Prediction
            st.markdown("""
            <div class='container-box'>
                <h3 style='color: #c2185b;'>Survival Predictions</h3>
                <p style='font-size: 22px; font-weight: bold;'>5-Year: {0:.2f}</p>
                <p style='font-size: 22px; font-weight: bold;'>10-Year: {1:.2f}</p>
            </div>
            """.format(survival_5yr, survival_10yr), unsafe_allow_html=True)

            st.success("✅ Patient record saved successfully!")

            # Results Overview
            st.markdown("""
            <h4 style='margin-top: 2rem; text-align: left; color: #c2185b;'>Results Overview</h4>
            """, unsafe_allow_html=True)

            # Layout of 3 columns
            col_chart, col_curve, col_text = st.columns(3)

            with col_chart:
                fig, ax = plt.subplots(figsize=(3,2))
                ax.bar(["5-Year", "10-Year"], [survival_5yr, survival_10yr], color="#FF69B4", width=0.5)
                ax.set_ylim(0,1)
                for idx, value in enumerate([survival_5yr, survival_10yr]):
                    ax.text(idx, value+0.03, f"{value:.2f}", ha='center', fontsize=9, fontweight='bold')
                ax.set_title("Survival Probability", fontsize=10)
                st.pyplot(fig)

            with col_curve:
                x = np.linspace(0, 10, 100)
                y = np.exp(-x/8)
                fig2, ax2 = plt.subplots(figsize=(3,2))
                ax2.plot(x, y, color="#ad1457", linewidth=2)
                ax2.set_xlabel("Years", fontsize=8)
                ax2.set_ylabel("Survival Probability", fontsize=8)
                ax2.set_title("Simulated Survival Curve", fontsize=10)
                st.pyplot(fig2)

            with col_text:
                risk_tag = ""
                recommendation = ""
                if survival_5yr > 0.8:
                    risk_tag = "High Survival Chance"
                    recommendation = "Continue standard monitoring."
                elif survival_5yr > 0.6:
                    risk_tag = "Moderate Survival Chance"
                    recommendation = "Consider more frequent follow-up."
                else:
                    risk_tag = "Low Survival Chance"
                    recommendation = "Consider aggressive treatment planning."

                st.markdown(f"""
                <div class='container-box' style='background-color: #fff0f5;'>
                    <h5 style='color:#c2185b;'>Risk Tag:</h5>
                    <p style='font-size:18px;font-weight:bold;'>{risk_tag}</p>
                    <h5 style='color:#c2185b;margin-top:15px;'>Recommendation:</h5>
                    <p style='font-size:16px;'>{recommendation}</p>
                </div>
                """, unsafe_allow_html=True)

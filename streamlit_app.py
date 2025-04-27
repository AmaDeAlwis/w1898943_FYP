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
st.text_input("Patient ID (Required)", key="patient_id")

# Previous Predictions
if st.session_state.patient_id:
    previous_records = list(collection.find({"patient_id": st.session_state.patient_id}))
    if previous_records:
        with st.expander("View Previous Predictions for this Patient ID"):
            for record in previous_records:
                st.write(f"Date: {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"5-Year Survival: {record['survival_5yr']:.2f}")
                st.write(f"10-Year Survival: {record['survival_10yr']:.2f}")
                st.markdown("---")

# Clinical Information
st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.text_input("Age", key="age")
    if st.session_state.age.strip():
        if not st.session_state.age.isdigit():
            st.warning("Age must be a number")
        elif int(st.session_state.age) < 20:
            st.warning("Age must be at least 20")

    st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="menopausal_status")
    st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="tumor_stage")

    st.text_input("Lymph Nodes Examined", key="lymph_nodes_examined")
    if st.session_state.lymph_nodes_examined.strip():
        if not st.session_state.lymph_nodes_examined.isdigit():
            st.warning("Lymph Nodes must be a number.")
        elif int(st.session_state.lymph_nodes_examined) < 0:
            st.warning("Lymph Nodes must be 0 or more.")

with col2:
    st.selectbox("ER Status", ["", "Positive", "Negative"], key="er_status")
    st.selectbox("PR Status", ["", "Positive", "Negative"], key="pr_status")
    st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"], key="her2_status")

# Treatment Information
st.markdown("<p class='section-title'>Treatment Information</p>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    st.selectbox("Chemotherapy", ["", "Yes", "No"], key="chemotherapy")
    st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"], key="surgery")

with col4:
    st.selectbox("Radiotherapy", ["", "Yes", "No"], key="radiotherapy")
    st.selectbox("Hormone Therapy", ["", "Yes", "No"], key="hormone_therapy")

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

# Prediction Logic
if predict_clicked:
    required_fields = [st.session_state.get(k, "") for k in field_keys]

    if not st.session_state.patient_id:
        st.warning("Please enter a Patient ID to save the record")
    elif "" in required_fields:
        st.warning("Please fill all required fields")
    elif not st.session_state.age.isdigit() or int(st.session_state.age) < 20:
        st.warning("Age must be a number and at least 20")
    elif not st.session_state.lymph_nodes_examined.isdigit() or int(st.session_state.lymph_nodes_examined) < 0:
        st.warning("Lymph Nodes must be a non-negative number")
    else:
        age = int(st.session_state.age)
        lymph_nodes_examined = int(st.session_state.lymph_nodes_examined)
        menopausal_status = 1 if st.session_state.menopausal_status == "Post-menopausal" else 0
        er_status = 1 if st.session_state.er_status == "Positive" else 0
        pr_status = 1 if st.session_state.pr_status == "Positive" else 0
        her2_val = st.session_state.her2_status
        her2_neutral = 1 if her2_val == "Neutral" else 0
        her2_loss = 1 if her2_val == "Loss" else 0
        her2_gain = 1 if her2_val == "Gain" else 0
        her2_undef = 1 if her2_val == "Undef" else 0
        chemotherapy = 1 if st.session_state.chemotherapy == "Yes" else 0
        radiotherapy = 1 if st.session_state.radiotherapy == "Yes" else 0
        hormone_therapy = 1 if st.session_state.hormone_therapy == "Yes" else 0
        surgery_conserving = 1 if st.session_state.surgery == "Breast-conserving" else 0
        surgery_mastectomy = 1 if st.session_state.surgery == "Mastectomy" else 0
        tumor_stage = int(st.session_state.tumor_stage)

        input_features = np.array([
            age, chemotherapy, er_status, hormone_therapy, menopausal_status,
            lymph_nodes_examined, pr_status, radiotherapy, tumor_stage,
            surgery_conserving, surgery_mastectomy, her2_gain,
            her2_loss, her2_neutral, her2_undef
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_features)
        x_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        graph_data = Data(x=x_tensor, edge_index=edge_index)

        with torch.no_grad():
            time_output, event_output = gcn_model(graph_data)
            survival_5yr = torch.sigmoid(time_output[0]).item()
            survival_10yr = torch.sigmoid(event_output[0]).item()

        # Save to MongoDB
        patient_data = {
            "patient_id": st.session_state.patient_id,
            "age": age,
            "menopausal_status": st.session_state.menopausal_status,
            "tumor_stage": tumor_stage,
            "lymph_nodes_examined": lymph_nodes_examined,
            "er_status": st.session_state.er_status,
            "pr_status": st.session_state.pr_status,
            "her2_status": st.session_state.her2_status,
            "chemotherapy": st.session_state.chemotherapy,
            "surgery": st.session_state.surgery,
            "radiotherapy": st.session_state.radiotherapy,
            "hormone_therapy": st.session_state.hormone_therapy,
            "timestamp": datetime.datetime.now(),
            "survival_5yr": survival_5yr,
            "survival_10yr": survival_10yr
        }
        collection.insert_one(patient_data)

        # Bar Chart + Risk Tag + Recommendation
        st.markdown("<h4 style='color: #c2185b;'>Results Overview</h4>", unsafe_allow_html=True)
        bar_col, tag_col, rec_col = st.columns([1,1,2])

        with bar_col:
            fig, ax = plt.subplots(figsize=(2.5,2))
            bars = ax.bar(["5-Year", "10-Year"], [survival_5yr, survival_10yr], color="#FF69B4", width=0.5)
            ax.set_ylim(0, 1)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2f}", ha='center', va='bottom', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)

        with tag_col:
            if survival_5yr >= 0.8:
                st.success("High Survival Chance")
            elif survival_5yr >= 0.6:
                st.warning("Moderate Survival Chance")
            else:
                st.error("Low Survival Chance")

        with rec_col:
            if survival_5yr >= 0.8:
                st.info("Continue standard monitoring.")
            elif survival_5yr >= 0.6:
                st.info("Consider more frequent follow-up.")
            else:
                st.info("Consider aggressive treatment planning.")

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
                st.write(f" Date: {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"- 5-Year Survival: {record['survival_5yr']:.2f}")
                st.write(f"- 10-Year Survival: {record['survival_10yr']:.2f}")
                st.markdown("---")

# Clinical Information
st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age", value=st.session_state.get("age", ""), key="age")
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

# Treatment Information
st.markdown("<p class='section-title'>Treatment Information</p>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    chemotherapy = st.selectbox("Chemotherapy", ["", "Yes", "No"],
                                index=0 if "chemotherapy" not in st.session_state else
                                ["", "Yes", "No"].index(st.session_state["chemotherapy"]),
                                key="chemotherapy")
    surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"],
                           index=0 if "surgery" not in st.session_state else
                           ["", "Breast-conserving", "Mastectomy"].index(st.session_state["surgery"]),
                           key="surgery")

with col4:
    radiotherapy = st.selectbox("Radiotherapy", ["", "Yes", "No"],
                                index=0 if "radiotherapy" not in st.session_state else
                                ["", "Yes", "No"].index(st.session_state["radiotherapy"]),
                                key="radiotherapy")
    hormone_therapy = st.selectbox("Hormone Therapy", ["", "Yes", "No"],
                                   index=0 if "hormone_therapy" not in st.session_state else
                                   ["", "Yes", "No"].index(st.session_state["hormone_therapy"]),
                                   key="hormone_therapy")

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

    if not patient_id or "" in required_fields or not st.session_state.age.isdigit():
        st.warning("Please fill all required fields correctly.")
    else:
        # Preprocessing
        age = int(st.session_state.age)
        lymph_nodes_examined = int(st.session_state.lymph_nodes_examined)
        menopausal_status = 1 if st.session_state.menopausal_status == "Post-menopausal" else 0
        er_status = 1 if st.session_state.er_status == "Positive" else 0
        pr_status = 1 if st.session_state.pr_status == "Positive" else 0
        her2_neutral = 1 if st.session_state.her2_status == "Neutral" else 0
        her2_loss = 1 if st.session_state.her2_status == "Loss" else 0
        her2_gain = 1 if st.session_state.her2_status == "Gain" else 0
        her2_undef = 1 if st.session_state.her2_status == "Undef" else 0
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
        collection.insert_one({
            "patient_id": patient_id,
            "timestamp": datetime.datetime.now(),
            "survival_5yr": survival_5yr,
            "survival_10yr": survival_10yr
        })

        st.success("✅ Patient record successfully saved!")

        # --- Visualization in a single row
        bar_col, risk_col, pie_col, text_col = st.columns([1, 1, 1, 2])

        with bar_col:
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.bar(["5-Year", "10-Year"], [survival_5yr, survival_10yr], color="#FF69B4", width=0.5)
            ax.set_ylim(0, 1)
            for i, v in enumerate([survival_5yr, survival_10yr]):
                ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)
            ax.set_ylabel("Probability")
            st.pyplot(fig)

        with risk_col:
            if survival_5yr > 0.80:
                risk_text = "High Survival Chance"
                risk_color = "green"
            elif survival_5yr > 0.60:
                risk_text = "Moderate Survival Chance"
                risk_color = "orange"
            else:
                risk_text = "Low Survival Chance"
                risk_color = "red"
            st.markdown(f"<h5 style='color: {risk_color}; text-align:center;'>{risk_text}</h5>", unsafe_allow_html=True)

        with pie_col:
            fig2, ax2 = plt.subplots(figsize=(2, 2))
            ax2.pie([survival_5yr, 1-survival_5yr], labels=["Survived", "Not Survived"], autopct='%1.0f%%', startangle=90, colors=["#90EE90", "#FF9999"])
            st.pyplot(fig2)

        with text_col:
            if survival_5yr > 0.80:
                st.success("Patient shows a high probability of 5-year survival. Continue standard monitoring.")
            elif survival_5yr > 0.60:
                st.warning("Patient shows moderate probability. Consider more frequent follow-up.")
            else:
                st.error("Patient shows low probability. Consider aggressive treatment planning.")

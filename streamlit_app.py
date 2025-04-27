import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import joblib
from torch_geometric.data import Data
from pymongo import MongoClient
import datetime
from gcn_model_class import SurvivalGNN

# Page config
st.set_page_config(page_title="Breast Cancer Survival Prediction", layout="wide")

# MongoDB Connection
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# Load model and scaler
gcn_model = SurvivalGNN(in_channels=15, out_channels_time=1, out_channels_event=1)
gcn_model.load_state_dict(torch.load(".streamlit/gcn_model.pt", map_location=torch.device("cpu")))
gcn_model.eval()
scaler = joblib.load("scaler.pkl")

# Initialize patient ID
if "patient_id" not in st.session_state:
    st.session_state["patient_id"] = ""

# Fields
field_keys = [
    "age", "menopausal_status", "tumor_stage", "lymph_nodes_examined",
    "er_status", "pr_status", "her2_status", "chemotherapy",
    "surgery", "radiotherapy", "hormone_therapy"
]

# --- Styling ---
st.markdown("""
    <style>
    h1 {
        text-align: center;
        color: #ad1457;
    }
    .section-title {
        font-size: 20px;
        font-weight: bold;
        color: #ad1457;
        margin-top: 2rem;
    }
    .stButton button {
        background-color: #ad1457 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>Breast Cancer Survival Prediction</h1>", unsafe_allow_html=True)

# --- Patient ID ---
st.markdown("<p class='section-title'>Patient Information</p>", unsafe_allow_html=True)
st.session_state.patient_id = st.text_input("Patient ID (Required)", value=st.session_state.get("patient_id", ""), key="patient_id")

# --- Clinical Data ---
col1, col2 = st.columns(2)
with col1:
    st.session_state.age = st.text_input("Age", value=st.session_state.get("age", ""), key="age")
    st.session_state.menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], index=0 if "menopausal_status" not in st.session_state else ["", "Pre-menopausal", "Post-menopausal"].index(st.session_state["menopausal_status"]), key="menopausal_status")
    st.session_state.tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], index=0 if "tumor_stage" not in st.session_state else ["", 1, 2, 3, 4].index(st.session_state["tumor_stage"]), key="tumor_stage")
    st.session_state.lymph_nodes_examined = st.text_input("Lymph Nodes Examined", value=st.session_state.get("lymph_nodes_examined", ""), key="lymph_nodes_examined")

with col2:
    st.session_state.er_status = st.selectbox("ER Status", ["", "Positive", "Negative"], index=0 if "er_status" not in st.session_state else ["", "Positive", "Negative"].index(st.session_state["er_status"]), key="er_status")
    st.session_state.pr_status = st.selectbox("PR Status", ["", "Positive", "Negative"], index=0 if "pr_status" not in st.session_state else ["", "Positive", "Negative"].index(st.session_state["pr_status"]), key="pr_status")
    st.session_state.her2_status = st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"], index=0 if "her2_status" not in st.session_state else ["", "Neutral", "Loss", "Gain", "Undef"].index(st.session_state["her2_status"]), key="her2_status")

# --- Treatment Section ---
st.markdown("<p class='section-title'>Treatment Information</p>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    st.session_state.chemotherapy = st.selectbox("Chemotherapy", ["", "Yes", "No"], index=0 if "chemotherapy" not in st.session_state else ["", "Yes", "No"].index(st.session_state["chemotherapy"]), key="chemotherapy")
    st.session_state.surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"], index=0 if "surgery" not in st.session_state else ["", "Breast-conserving", "Mastectomy"].index(st.session_state["surgery"]), key="surgery")

with col4:
    st.session_state.radiotherapy = st.selectbox("Radiotherapy", ["", "Yes", "No"], index=0 if "radiotherapy" not in st.session_state else ["", "Yes", "No"].index(st.session_state["radiotherapy"]), key="radiotherapy")
    st.session_state.hormone_therapy = st.selectbox("Hormone Therapy", ["", "Yes", "No"], index=0 if "hormone_therapy" not in st.session_state else ["", "Yes", "No"].index(st.session_state["hormone_therapy"]), key="hormone_therapy")

# --- Age instant validation ---
if st.session_state.get("age", ""):
    if not st.session_state.age.isdigit():
        st.warning("Age must be a number.")
    elif int(st.session_state.age) < 20:
        st.warning("Age must be at least 20.")

# --- Buttons ---
left, right = st.columns(2)
with left:
    if st.button("RESET"):
        for k in list(st.session_state.keys()):
            if k in field_keys + ["patient_id"]:
                del st.session_state[k]
        st.rerun()

with right:
    predict_clicked = st.button("PREDICT")

# --- Prediction Logic ---
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
        # --- Preprocess
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

        # --- Predict
        with torch.no_grad():
            time_output, event_output = gcn_model(graph_data)
            survival_5yr = torch.sigmoid(time_output[0]).item()
            survival_10yr = torch.sigmoid(event_output[0]).item()

        # --- Save to DB
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

        st.success("✅ Patient record successfully saved!")

        # --- Visualization ---
        st.markdown("<h4 style='color:#ad1457;'>Results Overview</h4>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        # Bar chart
        with col1:
            fig, ax = plt.subplots(figsize=(3, 3))
            bars = ax.bar(["5-Year", "10-Year"], [survival_5yr, survival_10yr], color="#ff69b4", width=0.5)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("Survival Probability", fontsize=14, fontweight='bold')
            for bar, value in zip(bars, [survival_5yr, survival_10yr]):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha='center', fontsize=10, fontweight='bold')
            st.pyplot(fig)

        # Middle - Risk & Recommendation
        with col2:
            if survival_5yr > 0.8:
                tag = "🟢 High Survival Chance"
                reco = "Patient shows high probability. Continue standard monitoring."
            elif survival_5yr > 0.6:
                tag = "🟡 Moderate Survival Chance"
                reco = "Consider more frequent follow-up."
            else:
                tag = "🔴 Low Survival Chance"
                reco = "Consider aggressive treatment planning."

            st.markdown(f"<div style='background-color: #ffe6e6; padding: 1rem; border-radius: 15px; height: 330px; display: flex; flex-direction: column; justify-content: center; align-items: center;'><h5 style='color:red;'>{tag}</h5><p style='color: #444;'>{reco}</p></div>", unsafe_allow_html=True)

        # Survival curve
        with col3:
            fig2, ax2 = plt.subplots(figsize=(3, 3))
            times = np.array([0, 0.5, 1.0])
            survival_probs = np.array([1.0, survival_5yr, survival_10yr])
            ax2.plot(times, survival_probs, marker='o', color='hotpink')
            ax2.set_ylim(0, 1)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Survival Probability")
            ax2.set_title("Estimated Survival Curve", fontsize=14, fontweight='bold')
            st.pyplot(fig2)

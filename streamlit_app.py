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

field_keys = [
    "age", "menopausal_status", "tumor_stage", "lymph_nodes_examined",
    "er_status", "pr_status", "her2_status", "chemotherapy",
    "surgery", "radiotherapy", "hormone_therapy"
]

# CSS Styling
st.markdown("""
<style>
h1 {color: #ad1457 !important; text-align: center; font-weight: bold;}
.section-title {font-size: 20px; font-weight: bold; margin-top: 2rem; margin-bottom: 0.5rem; color: #ad1457;}
.stButton button {background-color: #ad1457 !important; color: white !important; font-weight: bold; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1> Breast Cancer Survival Prediction </h1>", unsafe_allow_html=True)

# --- Patient ID ---
patient_id = st.text_input("Patient ID (Required)", value=st.session_state["patient_id"], key="patient_id")

# --- Clinical Form ---
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age", value=st.session_state.get("age", ""), key="age")
    if st.session_state.age:
        if not st.session_state.age.isdigit() or int(st.session_state.age) < 20:
            st.warning("Age must be a number and at least 20")

    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="menopausal_status")
    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="tumor_stage")
    lymph_nodes_examined = st.text_input("Lymph Nodes Examined", value=st.session_state.get("lymph_nodes_examined", ""), key="lymph_nodes_examined")

with col2:
    er_status = st.selectbox("ER Status", ["", "Positive", "Negative"], key="er_status")
    pr_status = st.selectbox("PR Status", ["", "Positive", "Negative"], key="pr_status")
    her2_status = st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"], key="her2_status")
    chemotherapy = st.selectbox("Chemotherapy", ["", "Yes", "No"], key="chemotherapy")
    surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"], key="surgery")
    radiotherapy = st.selectbox("Radiotherapy", ["", "Yes", "No"], key="radiotherapy")
    hormone_therapy = st.selectbox("Hormone Therapy", ["", "Yes", "No"], key="hormone_therapy")

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

# --- Predict ---
if predict_clicked:
    required_fields = [st.session_state.get(k, "") for k in field_keys]
    if not patient_id:
        st.warning("Please enter a Patient ID")
    elif "" in required_fields:
        st.warning("Please fill all required fields")
    else:
        age = int(st.session_state.age)
        lymph_nodes_examined = int(st.session_state.lymph_nodes_examined)
        menopausal_status = 1 if st.session_state.menopausal_status == "Post-menopausal" else 0
        er_status = 1 if st.session_state.er_status == "Positive" else 0
        pr_status = 1 if st.session_state.pr_status == "Positive" else 0
        her2_vals = {
            "Neutral": [1, 0, 0, 0],
            "Loss": [0, 1, 0, 0],
            "Gain": [0, 0, 1, 0],
            "Undef": [0, 0, 0, 1],
        }
        her2_encoded = her2_vals.get(st.session_state.her2_status, [0, 0, 0, 0])

        input_features = np.array([
            age, 1 if st.session_state.chemotherapy == "Yes" else 0, er_status, 1 if st.session_state.hormone_therapy == "Yes" else 0,
            menopausal_status, lymph_nodes_examined, pr_status, 1 if st.session_state.radiotherapy == "Yes" else 0,
            int(st.session_state.tumor_stage),
            1 if st.session_state.surgery == "Breast-conserving" else 0,
            1 if st.session_state.surgery == "Mastectomy" else 0,
            *her2_encoded
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_features)
        x_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        graph_data = Data(x=x_tensor, edge_index=edge_index)

        with torch.no_grad():
            time_output, event_output = gcn_model(graph_data)
            survival_5yr = torch.sigmoid(time_output[0]).item()
            survival_10yr = torch.sigmoid(event_output[0]).item()

        # --- Display White Box First ---
        st.markdown(f"""
            <div style='background-color: #ffffff; padding: 2rem; border-radius: 20px;
                        box-shadow: 0 4px 12px rgba(220, 20, 60, 0.15);
                        width: 100%; max-width: 600px; margin: auto; text-align: center;'>
                <h3 style='color: #c2185b;'>Survival Predictions</h3>
                <p style='font-size: 22px; font-weight: bold; color: #004d40;'>5-Year Survival Probability: {survival_5yr:.2f}</p>
                <p style='font-size: 22px; font-weight: bold; color: #004d40;'>10-Year Survival Probability: {survival_10yr:.2f}</p>
            </div>
        """, unsafe_allow_html=True)

        # --- Save Record ---
        patient_data = {
            "patient_id": patient_id,
            "timestamp": datetime.datetime.now(),
            "survival_5yr": survival_5yr,
            "survival_10yr": survival_10yr,
            **{k: st.session_state[k] for k in field_keys}
        }
        collection.insert_one(patient_data)
        st.success("✅ Patient record successfully saved!")

        # --- Visualizations ---
        st.markdown("<h4 style='text-align: center; color: #c2185b;'>Results Overview</h4>", unsafe_allow_html=True)
        col_v1, col_v2, col_v3 = st.columns([1, 1, 1])

        with col_v1:
            fig, ax = plt.subplots(figsize=(3, 2))
            bars = ax.bar(["5-Year", "10-Year"], [survival_5yr, survival_10yr], color="#FF69B4", width=0.5)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("Survival Probabilities", fontsize=10)
            for bar, value in zip(bars, [survival_5yr, survival_10yr]):
                ax.text(bar.get_x() + bar.get_width()/2, value + 0.02, f"{value:.2f}", ha='center', fontsize=8)
            st.pyplot(fig)

        with col_v2:
            # Survival Curve (simple fake)
            months = np.linspace(0, 120, 100)
            survival_curve = np.clip(1 - (months/120) * (1-survival_5yr), 0, 1)
            fig2, ax2 = plt.subplots(figsize=(3, 2))
            ax2.plot(months, survival_curve, color="#FF69B4")
            ax2.set_xlabel("Months")
            ax2.set_ylabel("Probability")
            ax2.set_title("Survival Curve", fontsize=10)
            st.pyplot(fig2)

        with col_v3:
            if survival_5yr > 0.8:
                st.success("High Survival Chance\n\nPatient shows high probability. Continue standard monitoring.")
            elif survival_5yr > 0.6:
                st.warning("Moderate Survival Chance\n\nPatient shows moderate probability. Consider more frequent follow-up.")
            else:
                st.error("Low Survival Chance\n\nPatient shows low probability. Consider aggressive treatment planning.")

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

# Load model and scaler
gcn_model = SurvivalGNN(in_channels=15, out_channels_time=1, out_channels_event=1)
gcn_model.load_state_dict(torch.load(".streamlit/gcn_model.pt", map_location=torch.device("cpu")))
gcn_model.eval()
scaler = joblib.load("scaler.pkl")

# MongoDB connection
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

# --- Page Title ---
st.markdown("""
<style>
h1 {
    color: #ad1457 !important;
    text-align: center;
    font-weight: bold;
}
.section-title {
    font-size: 22px;
    font-weight: bold;
    color: #ad1457;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
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

# --- Patient Information Section ---
st.markdown("<p class='section-title'>Patient Information</p>", unsafe_allow_html=True)
patient_id = st.text_input("Patient ID (Required)", value=st.session_state["patient_id"], key="patient_id")

# --- Clinical and Treatment Info ---
col1, col2 = st.columns(2)
with col1:
    st.session_state.age = st.text_input("Age", value=st.session_state.get("age", ""), key="age")
    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="menopausal_status")
    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="tumor_stage")
    lymph_nodes_examined = st.text_input("Lymph Nodes Examined", value=st.session_state.get("lymph_nodes_examined", ""), key="lymph_nodes_examined")

with col2:
    er_status = st.selectbox("ER Status", ["", "Positive", "Negative"], key="er_status")
    pr_status = st.selectbox("PR Status", ["", "Positive", "Negative"], key="pr_status")
    her2_status = st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"], key="her2_status")

# --- Treatment Section ---
st.markdown("<p class='section-title'>Treatment Information</p>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    chemotherapy = st.selectbox("Chemotherapy", ["", "Yes", "No"], key="chemotherapy")
    surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"], key="surgery")

with col4:
    radiotherapy = st.selectbox("Radiotherapy", ["", "Yes", "No"], key="radiotherapy")
    hormone_therapy = st.selectbox("Hormone Therapy", ["", "Yes", "No"], key="hormone_therapy")

# --- Buttons ---
col_reset, col_predict = st.columns(2)
with col_reset:
    if st.button("RESET"):
        for k in list(st.session_state.keys()):
            if k in field_keys + ["patient_id"]:
                del st.session_state[k]
        st.experimental_rerun()

with col_predict:
    predict_clicked = st.button("PREDICT")

# --- Prediction Logic ---
if predict_clicked:
    required_fields = [st.session_state.get(k, "") for k in field_keys]

    if not patient_id:
        st.warning("Please enter a Patient ID to save the record")
    elif "" in required_fields:
        st.warning("Please fill all required fields")
    elif not st.session_state.age.isdigit() or int(st.session_state.age) < 20:
        st.warning("Age must be at least 20")
    elif not st.session_state.lymph_nodes_examined.isdigit() or int(st.session_state.lymph_nodes_examined) < 0:
        st.warning("Lymph Nodes must be a non-negative number")
    else:
        age = int(st.session_state.age)
        lymph_nodes_examined = int(st.session_state.lymph_nodes_examined)
        menopausal_status_val = 1 if st.session_state.menopausal_status == "Post-menopausal" else 0
        er_status_val = 1 if st.session_state.er_status == "Positive" else 0
        pr_status_val = 1 if st.session_state.pr_status == "Positive" else 0
        her2_neutral = 1 if st.session_state.her2_status == "Neutral" else 0
        her2_loss = 1 if st.session_state.her2_status == "Loss" else 0
        her2_gain = 1 if st.session_state.her2_status == "Gain" else 0
        her2_undef = 1 if st.session_state.her2_status == "Undef" else 0
        chemotherapy_val = 1 if st.session_state.chemotherapy == "Yes" else 0
        radiotherapy_val = 1 if st.session_state.radiotherapy == "Yes" else 0
        hormone_therapy_val = 1 if st.session_state.hormone_therapy == "Yes" else 0
        surgery_conserving = 1 if st.session_state.surgery == "Breast-conserving" else 0
        surgery_mastectomy = 1 if st.session_state.surgery == "Mastectomy" else 0
        tumor_stage_val = int(st.session_state.tumor_stage)

        input_features = np.array([
            age, chemotherapy_val, er_status_val, hormone_therapy_val, menopausal_status_val,
            lymph_nodes_examined, pr_status_val, radiotherapy_val, tumor_stage_val,
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

        # --- Save to MongoDB ---
        patient_data = {
            "patient_id": patient_id,
            "age": age,
            "menopausal_status": st.session_state.menopausal_status,
            "tumor_stage": tumor_stage_val,
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

        st.success("\n✅ Patient record successfully saved!")

        # --- Results Overview Section ---
        st.markdown("<h2 style='color:#ad1457;'>Results Overview</h2>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])

        # Column 1: Bar Chart
        with col1:
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            bars = ax1.bar(["5-Year", "10-Year"], [survival_5yr, survival_10yr], color="#FF69B4")
            ax1.set_ylim(0, 1)
            ax1.set_ylabel("Probability")
            for bar, value in zip(bars, [survival_5yr, survival_10yr]):
                ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha='center', fontsize=10, fontweight='bold')
            st.pyplot(fig1)

        # Column 2: Risk, Recommendation, Pie Charts
        with col2:
            # Risk Tag
            if survival_5yr > 0.80:
                risk_text = "High Survival Chance"
                risk_color = "#d4edda"
            elif survival_5yr > 0.60:
                risk_text = "Moderate Survival Chance"
                risk_color = "#fff3cd"
            else:
                risk_text = "Low Survival Chance"
                risk_color = "#f8d7da"
            st.markdown(f"""
                <div style='background-color:{risk_color}; padding:1rem; border-radius:10px; font-weight:bold; text-align:center;'>
                    {risk_text}
                </div>
            """, unsafe_allow_html=True)

            st.info(
                """
                Patient shows {} probability. {}
                """.format(
                    "high" if survival_5yr > 0.80 else "moderate" if survival_5yr > 0.60 else "low",
                    "Continue standard monitoring." if survival_5yr > 0.80 else "Consider more frequent follow-up." if survival_5yr > 0.60 else "Consider aggressive treatment planning."
                )
            )

            # Pie Charts (Side by side)
            pie1, pie2 = st.columns(2)
            with pie1:
                fig5, ax5 = plt.subplots(figsize=(1.5, 1.5))
                ax5.pie([survival_5yr, 1-survival_5yr], labels=["Survived", "Not Survived"], autopct='%1.0f%%', colors=["#90EE90", "#FF7F7F"])
                st.pyplot(fig5)

            with pie2:
                fig6, ax6 = plt.subplots(figsize=(1.5, 1.5))
                ax6.pie([survival_10yr, 1-survival_10yr], labels=["Survived", "Not Survived"], autopct='%1.0f%%', colors=["#90EE90", "#FF7F7F"])
                st.pyplot(fig6)

        # Column 3: Survival Curve
        with col3:
            fig2, ax2 = plt.subplots(figsize=(3, 3))
            times = np.linspace(0, 1, 100)
            survival_probs = np.exp(-2 * times)
            ax2.plot(times, survival_probs, color="#FF69B4", linewidth=2)
            ax2.set_title("Survival Curve")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Survival Probability")
            ax2.set_ylim(0, 1)
            st.pyplot(fig2)

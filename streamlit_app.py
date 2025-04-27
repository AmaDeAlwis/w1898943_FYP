import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import joblib
from torch_geometric.data import Data
from pymongo import MongoClient
import datetime
from gcn_model_class import SurvivalGNN

# --- Streamlit Page Config ---
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# --- Load Model and Scaler ---
gcn_model = SurvivalGNN(in_channels=15, out_channels_time=1, out_channels_event=1)
gcn_model.load_state_dict(torch.load(".streamlit/gcn_model.pt", map_location=torch.device("cpu")))
gcn_model.eval()
scaler = joblib.load("scaler.pkl")

# --- MongoDB Connection ---
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# --- CSS Styling ---
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

# --- Initialize Session State ---
if "patient_id" not in st.session_state:
    st.session_state["patient_id"] = ""

field_keys = [
    "age", "menopausal_status", "tumor_stage", "lymph_nodes_examined",
    "er_status", "pr_status", "her2_status", "chemotherapy",
    "surgery", "radiotherapy", "hormone_therapy"
]

# --- Title ---
st.markdown("<h1>Breast Cancer Survival Prediction</h1>", unsafe_allow_html=True)

# --- Patient Info ---
st.markdown("<p class='section-title'>Patient Information</p>", unsafe_allow_html=True)
patient_id = st.text_input("Patient ID (Required)", value=st.session_state["patient_id"], key="patient_id")

if patient_id:
    previous_records = list(collection.find({"patient_id": patient_id}))
    if previous_records:
        with st.expander("View Previous Predictions for this Patient ID"):
            for record in previous_records:
                st.write(f"Date: {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"5-Year Survival: {record['survival_5yr']:.2f}")
                st.write(f"10-Year Survival: {record['survival_10yr']:.2f}")
                st.markdown("---")

# --- Clinical Info ---
st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.text_input("Age", value=st.session_state.get("age", ""), key="age")
    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="menopausal_status")
    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="tumor_stage")
    lymph_nodes_examined = st.text_input("Lymph Nodes Examined", value=st.session_state.get("lymph_nodes_examined", ""), key="lymph_nodes_examined")

with col2:
    er_status = st.selectbox("ER Status", ["", "Positive", "Negative"], key="er_status")
    pr_status = st.selectbox("PR Status", ["", "Positive", "Negative"], key="pr_status")
    her2_status = st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"], key="her2_status")

# --- Treatment Info ---
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
        st.rerun()

with col_predict:
    predict_clicked = st.button("PREDICT")

# --- Prediction Logic ---
if predict_clicked:
    if not patient_id:
        st.warning("Please enter a Patient ID to save the record.")
    elif "" in [st.session_state.get(k, "") for k in field_keys]:
        st.warning("Please fill all required fields.")
    elif not st.session_state.age.isdigit() or int(st.session_state.age) < 20:
        st.warning("Age must be at least 20.")
    elif not st.session_state.lymph_nodes_examined.isdigit():
        st.warning("Lymph Nodes must be a valid non-negative number.")
    else:
        # Preprocessing
        age = int(st.session_state.age)
        lymph_nodes_examined = int(st.session_state.lymph_nodes_examined)
        menopausal_status = 1 if st.session_state.menopausal_status == "Post-menopausal" else 0
        er_status = 1 if st.session_state.er_status == "Positive" else 0
        pr_status = 1 if st.session_state.pr_status == "Positive" else 0
        her2_vals = st.session_state.her2_status
        her2_neutral = 1 if her2_vals == "Neutral" else 0
        her2_loss = 1 if her2_vals == "Loss" else 0
        her2_gain = 1 if her2_vals == "Gain" else 0
        her2_undef = 1 if her2_vals == "Undef" else 0
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

        x_scaled = scaler.transform(features)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        graph_data = Data(x=x_tensor, edge_index=torch.tensor([[0], [0]]))

        with torch.no_grad():
            time_out, event_out = gcn_model(graph_data)
            survival_5yr = torch.sigmoid(time_out[0]).item()
            survival_10yr = torch.sigmoid(event_out[0]).item()

        # --- Display Probabilities First ---
        st.markdown(f"""
            <div style='background-color: #ffffff; padding: 2rem; border-radius: 15px; 
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1); width: 70%; margin: auto;'>
                <h3 style='color: #c2185b; text-align:center;'>Survival Predictions</h3>
                <p style='font-size:22px; font-weight:bold; text-align:center;'>5-Year Survival Probability: {survival_5yr:.2f}</p>
                <p style='font-size:22px; font-weight:bold; text-align:center;'>10-Year Survival Probability: {survival_10yr:.2f}</p>
            </div>
        """, unsafe_allow_html=True)

        # --- Save to MongoDB ---
        collection.insert_one({
            "patient_id": patient_id,
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
        })

        st.success("✅ Patient record successfully saved!")

        # --- Visual Overview ---
        st.markdown("<h4 style='text-align: center; color: #c2185b;'>Results Overview</h4>", unsafe_allow_html=True)

        chart_col, risk_col, reco_col = st.columns(3)

        with chart_col:
            fig, ax = plt.subplots(figsize=(3,2))
            bars = ax.bar(["5-Year", "10-Year"], [survival_5yr, survival_10yr], color="#FF69B4", width=0.5)
            ax.set_ylim(0, 1)
            for bar, val in zip(bars, [survival_5yr, survival_10yr]):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}", ha='center', fontsize=9)
            st.pyplot(fig)

        with risk_col:
            if survival_5yr > 0.8:
                st.success("High Survival Chance")
            elif survival_5yr > 0.6:
                st.warning("Moderate Survival Chance")
            else:
                st.error("Low Survival Chance")

        with reco_col:
            if survival_5yr > 0.8:
                st.info("Patient shows a high probability of 5-year survival. Continue standard monitoring.")
            elif survival_5yr > 0.6:
                st.info("Patient shows moderate probability. Consider more frequent follow-up.")
            else:
                st.info("Patient shows low probability. Consider aggressive treatment planning.")

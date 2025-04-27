import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import joblib
from torch_geometric.data import Data
from pymongo import MongoClient
import datetime
from gcn_model_class import SurvivalGNN

# --- Page Config ---
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# --- Load model and scaler ---
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
h1 { color: #ad1457; text-align: center; font-weight: bold; }
.section-title { font-size: 20px; font-weight: bold; color: #ad1457; margin-top: 2rem; }
.stButton button { background-color: #ad1457 !important; color: white !important; font-weight: bold; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Initialize ---
field_keys = [
    "age", "menopausal_status", "tumor_stage", "lymph_nodes_examined",
    "er_status", "pr_status", "her2_status", "chemotherapy",
    "surgery", "radiotherapy", "hormone_therapy"
]
if "patient_id" not in st.session_state:
    st.session_state["patient_id"] = ""

# --- Patient Information ---
st.markdown("<h1>Breast Cancer Survival Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>Patient Information</p>", unsafe_allow_html=True)

patient_id = st.text_input("Patient ID (Required)", value=st.session_state["patient_id"], key="patient_id")

# --- Clinical Information ---
st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age", value=st.session_state.get("age", ""), key="age")
    if age.strip():
        if not age.isdigit():
            st.warning("Age must be a number.")
        elif int(age) < 20:
            st.warning("Age must be at least 20.")
    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="menopausal_status")
    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="tumor_stage")
    lymph_nodes_examined = st.text_input("Lymph Nodes Examined", value=st.session_state.get("lymph_nodes_examined", ""), key="lymph_nodes_examined")
    if lymph_nodes_examined.strip() and not lymph_nodes_examined.isdigit():
        st.warning("Lymph Nodes must be a non-negative number.")

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
        for k in list(st.session_state.keys()):
            if k in field_keys + ["patient_id"]:
                del st.session_state[k]
        st.rerun()
with right:
    predict_clicked = st.button("PREDICT")

# --- Prediction Logic ---
if predict_clicked:
    required_fields = [st.session_state.get(k, "") for k in field_keys]

    if not patient_id:
        st.warning("Please enter a Patient ID to save the record")
    elif "" in required_fields:
        st.warning("Please fill all required fields")
    elif not st.session_state.age.isdigit() or int(st.session_state.age) < 20:
        st.warning("Age must be a number and at least 20")
    elif not st.session_state.lymph_nodes_examined.isdigit() or int(st.session_state.lymph_nodes_examined) < 0:
        st.warning("Lymph Nodes must be a non-negative number")
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

        # --- Display Survival Prediction ---
        st.markdown(f"""
            <div style='text-align: center; background: #fff; padding: 2rem; border-radius: 15px;'>
                <h2 style='color: #ad1457;'>Survival Predictions</h2>
                <p style='font-size: 22px; font-weight: bold;'>5-Year Survival Probability: {survival_5yr:.2f}</p>
                <p style='font-size: 22px; font-weight: bold;'>10-Year Survival Probability: {survival_10yr:.2f}</p>
            </div>
        """, unsafe_allow_html=True)

        # --- Save to MongoDB ---
        patient_data = {
            "patient_id": patient_id,
            "timestamp": datetime.datetime.now(),
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
            "survival_5yr": survival_5yr,
            "survival_10yr": survival_10yr
        }
        collection.insert_one(patient_data)

        st.success("Patient record successfully saved!")

        # --- Results Overview ---
        st.markdown("<h3 style='text-align: left; color: #ad1457;'>Results Overview</h3>", unsafe_allow_html=True)
        chart_col, risk_col, recommendation_col = st.columns(3)

        with chart_col:
            fig, ax = plt.subplots(figsize=(2.5, 2.5))
            bars = ax.bar(["5-Year", "10-Year"], [survival_5yr, survival_10yr], color="#FF69B4", width=0.5)
            ax.set_ylim(0, 1)
            for bar, value in zip(bars, [survival_5yr, survival_10yr]):
                ax.text(bar.get_x() + bar.get_width()/2, value + 0.02, f"{value:.2f}", ha='center', fontsize=8)
            ax.set_ylabel("Probability")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)

        with risk_col:
            if survival_5yr > 0.80:
                st.success("High Survival Chance")
            elif survival_5yr > 0.60:
                st.warning("Moderate Survival Chance")
            else:
                st.error("Low Survival Chance")

        with recommendation_col:
            if survival_5yr > 0.80:
                st.info("Patient shows a high probability. Continue standard monitoring.")
            elif survival_5yr > 0.60:
                st.info("Patient shows moderate probability. Consider more frequent follow-up.")
            else:
                st.info("Patient shows low probability. Consider aggressive treatment planning.")

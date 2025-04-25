import streamlit as st
import torch
import numpy as np
import joblib
from torch_geometric.data import Data
from pymongo import MongoClient
import datetime
from gcn_model_class import SurvivalGNN

# Configure app
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# Load model and scaler
gcn_model = SurvivalGNN(in_channels=15, out_channels_time=1, out_channels_event=1)
gcn_model.load_state_dict(torch.load(".streamlit/gcn_model.pt", map_location=torch.device('cpu')))
gcn_model.eval()
scaler = joblib.load("scaler.pkl")

# Connect to MongoDB
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# Initialize field keys
field_keys = [
    "age", "menopausal_status", "tumor_stage", "lymph_nodes_examined",
    "er_status", "pr_status", "her2_status", "chemotherapy",
    "surgery", "radiotherapy", "hormone_therapy"
]

# --- Custom CSS ---
st.markdown("""
<style>
h1 {
    text-align: center;
    color: #FFFFFF;
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

st.markdown("<h1> Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)

# Form
with st.form("input_form"):
    # Section: Clinical Data
    st.markdown("<p class='section-title'>Clinical Data</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.text_input("Age", key="age")
        if age and (not age.isdigit() or int(age) < 20):
            st.markdown("<span style='color:red;'>⚠️ Age must be a number and at least 20.</span>", unsafe_allow_html=True)
        menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="menopausal_status")
        tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="tumor_stage")
        lymph_nodes_examined = st.text_input("Lymph Nodes Examined", key="lymph_nodes_examined")
        if lymph_nodes_examined and (not lymph_nodes_examined.isdigit() or int(lymph_nodes_examined) < 0):
            st.markdown("<span style='color:red;'>⚠️ Lymph Nodes Examined must be a non-negative number.</span>", unsafe_allow_html=True)

    with col2:
        er_status = st.selectbox("ER Status", ["", "Positive", "Negative"], key="er_status")
        pr_status = st.selectbox("PR Status", ["", "Positive", "Negative"], key="pr_status")
        her2_status = st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"], key="her2_status")

    # Section: Treatment Data
    st.markdown("<p class='section-title'>Treatment Data</p>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        chemotherapy = st.selectbox("Chemotherapy", ["", "Yes", "No"], key="chemotherapy")
        surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"], key="surgery")
    with col4:
        radiotherapy = st.selectbox("Radiotherapy", ["", "Yes", "No"], key="radiotherapy")
        hormone_therapy = st.selectbox("Hormone Therapy", ["", "Yes", "No"], key="hormone_therapy")

    # Buttons
    left, right = st.columns([1, 1])
    with left:
        reset = st.form_submit_button("RESET")
    with right:
        predict = st.form_submit_button("PREDICT")

# Safe RESET logic (no rerun crash)
if reset:
    for k in field_keys:
        if k in st.session_state:
            st.session_state[k] = ""
    st.experimental_set_query_params(reset="true")
    st.success("Form has been reset.")

# Prediction logic
required_fields = [st.session_state.get(k, "") for k in field_keys]
if predict:
    if "" in required_fields:
        st.warning(" Please fill in all the required fields.")
    else:
        st.warning(" Lymph Nodes Examined must be a non-negative number.")
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

        st.markdown(f"""
            <div style='background-color: #ffffff; padding: 2rem; border-radius: 20px;
                 box-shadow: 0 4px 12px rgba(220, 20, 60, 0.15); margin-top: 2rem;
                 text-align: center; width: 90%; margin-left: auto; margin-right: auto;'>
                <h3 style='color: #c2185b;'> Survival Predictions</h3>
                <p style='font-size: 22px; font-weight: bold; color: #880e4f;'>🩺 5-Year: <span style="color:#d81b60;">{survival_5yr:.2f}</span></p>
                <p style='font-size: 22px; font-weight: bold; color: #880e4f;'>🩺 10-Year: <span style="color:#d81b60;">{survival_10yr:.2f}</span></p>
            </div>
        """, unsafe_allow_html=True)

        patient_data = {key: st.session_state.get(key) for key in field_keys}
        patient_data.update({
            "timestamp": datetime.datetime.now(),
            "survival_5yr": survival_5yr,
            "survival_10yr": survival_10yr
        })
        collection.insert_one(patient_data)

        st.success(" Patient record successfully saved to MongoDB Atlas.")

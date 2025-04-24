import streamlit as st
import torch
import numpy as np
import joblib
from torch_geometric.data import Data
from pymongo import MongoClient
import datetime
from gcn_model_class import SurvivalGNN

st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# Load model and scaler
gcn_model = SurvivalGNN(in_channels=15, out_channels_time=1, out_channels_event=1)
gcn_model.load_state_dict(torch.load(".streamlit/gcn_model.pt", map_location=torch.device('cpu')))
gcn_model.eval()
scaler = joblib.load("scaler.pkl")

# MongoDB connection
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# Initialize session state for all inputs
fields = [
    "age", "menopausal_status", "tumor_stage", "lymph_nodes_examined",
    "er_status", "pr_status", "her2_status", "chemotherapy",
    "surgery", "radiotherapy", "hormone_therapy"
]
def initialize_fields():
    for field in fields:
        if field not in st.session_state:
            st.session_state[field] = ""
initialize_fields()

# --- Handle RESET ---
if st.query_params.get("reset") == "true":
    for field in fields:
        st.session_state[field] = ""
    st.query_params.clear()
    st.experimental_rerun()

# --- Custom CSS ---
st.markdown("""
<style>
h1 { text-align: center; color: #FFFFFF; }
.section-title { font-size: 20px; font-weight: bold; margin-top: 2rem; margin-bottom: 0.5rem; color: #ad1457; }
button[kind="primary"] {
    background-color: #ad1457 !important;
    color: white !important;
    font-weight: bold !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    margin-top: 1rem !important;
    cursor: pointer !important;
}
input, select, textarea { border-radius: 10px !important; cursor: pointer !important; }
</style>
""", unsafe_allow_html=True)

# --- UI ---
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown("<h1> Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Fill in the details below to generate predictions and insights.</p>", unsafe_allow_html=True)

with st.form("input_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        age = st.text_input("Age", value=st.session_state.age)
        menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], index=["", "Pre-menopausal", "Post-menopausal"].index(st.session_state.menopausal_status))
        tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], index=["", 1, 2, 3, 4].index(st.session_state.tumor_stage) if st.session_state.tumor_stage in [1,2,3,4] else 0)
        lymph_nodes_examined = st.text_input("Lymph Nodes Examined", value=st.session_state.lymph_nodes_examined)
    with col2:
        er_status = st.selectbox("ER Status", ["", "Positive", "Negative"], index=["", "Positive", "Negative"].index(st.session_state.er_status))
        pr_status = st.selectbox("PR Status", ["", "Positive", "Negative"], index=["", "Positive", "Negative"].index(st.session_state.pr_status))
        her2_status = st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"], index=["", "Neutral", "Loss", "Gain", "Undef"].index(st.session_state.her2_status))

    col3, col4 = st.columns(2)
    with col3:
        chemotherapy = st.selectbox("Chemotherapy", ["", "Yes", "No"], index=["", "Yes", "No"].index(st.session_state.chemotherapy))
        surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"], index=["", "Breast-conserving", "Mastectomy"].index(st.session_state.surgery))
    with col4:
        radiotherapy = st.selectbox("Radiotherapy", ["", "Yes", "No"], index=["", "Yes", "No"].index(st.session_state.radiotherapy))
        hormone_therapy = st.selectbox("Hormone Therapy", ["", "Yes", "No"], index=["", "Yes", "No"].index(st.session_state.hormone_therapy))

    colA, colB = st.columns(2)
    with colA:
        reset = st.form_submit_button("RESET")
    with colB:
        predict = st.form_submit_button("PREDICT")

# Save input values
st.session_state.age = age
st.session_state.menopausal_status = menopausal_status
st.session_state.tumor_stage = tumor_stage
st.session_state.lymph_nodes_examined = lymph_nodes_examined
st.session_state.er_status = er_status
st.session_state.pr_status = pr_status
st.session_state.her2_status = her2_status
st.session_state.chemotherapy = chemotherapy
st.session_state.surgery = surgery
st.session_state.radiotherapy = radiotherapy
st.session_state.hormone_therapy = hormone_therapy

if reset:
    st.query_params["reset"] = "true"
    st.experimental_rerun()

if predict:
    # Validation
    missing = [k for k in fields if not st.session_state.get(k)]
    if missing:
        st.warning("Please fill all the required fields.")
    else:
        try:
            input_features = np.array([
                float(age),
                1 if chemotherapy == "Yes" else 0,
                1 if er_status == "Positive" else 0,
                1 if hormone_therapy == "Yes" else 0,
                1 if menopausal_status == "Post-menopausal" else 0,
                float(lymph_nodes_examined),
                1 if pr_status == "Positive" else 0,
                1 if radiotherapy == "Yes" else 0,
                int(tumor_stage),
                1 if surgery == "Breast-conserving" else 0,
                1 if surgery == "Mastectomy" else 0,
                1 if her2_status == "Gain" else 0,
                1 if her2_status == "Loss" else 0,
                1 if her2_status == "Neutral" else 0,
                1 if her2_status == "Undef" else 0
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
                    <div style='margin-bottom: 1.5rem;'>
                        <p style='font-size: 22px; font-weight: bold; color: #880e4f;'>🩺 5-Year Survival Probability:
                            <span style="color:#d81b60;">{survival_5yr:.2f}</span></p>
                    </div>
                    <div>
                        <p style='font-size: 22px; font-weight: bold; color: #880e4f;'>🩺 10-Year Survival Probability:
                            <span style="color:#d81b60;">{survival_10yr:.2f}</span></p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            patient_data = {
                "timestamp": datetime.datetime.now(),
                "age": float(age),
                "menopausal_status": menopausal_status,
                "tumor_stage": int(tumor_stage),
                "lymph_nodes_examined": float(lymph_nodes_examined),
                "er_status": er_status,
                "pr_status": pr_status,
                "her2_status": her2_status,
                "chemotherapy": chemotherapy,
                "radiotherapy": radiotherapy,
                "hormone_therapy": hormone_therapy,
                "surgery": surgery,
                "survival_5yr": survival_5yr,
                "survival_10yr": survival_10yr
            }
            collection.insert_one(patient_data)

            st.markdown("""
                <div style='margin-top: 1.5rem; background-color: #fce4ec; padding: 1rem;
                            border-radius: 15px; color: #880e4f; font-weight: bold;
                            text-align: center;'>
                    Patient prediction record successfully saved to MongoDB Atlas.
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error("Something went wrong. Please check your inputs.")

st.markdown("</div>", unsafe_allow_html=True)

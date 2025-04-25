import streamlit as st
import torch
import numpy as np
import joblib
from torch_geometric.data import Data
from pymongo import MongoClient
import datetime
from gcn_model_class import SurvivalGNN

# --- Configuration ---
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# --- Load model & scaler ---
gcn_model = SurvivalGNN(in_channels=15, out_channels_time=1, out_channels_event=1)
gcn_model.load_state_dict(torch.load(".streamlit/gcn_model.pt", map_location=torch.device("cpu")))
gcn_model.eval()
scaler = joblib.load("scaler.pkl")

# --- MongoDB Connection ---
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# --- Field Keys ---
field_keys = [
    "age", "menopausal_status", "tumor_stage", "lymph_nodes_examined",
    "er_status", "pr_status", "her2_status", "chemotherapy",
    "surgery", "radiotherapy", "hormone_therapy"
]

# --- Check for query param to reset ---
if "reset" in st.query_params:
    for k in field_keys:
        if k in st.session_state:
            del st.session_state[k]
    st.query_params.clear()

# --- CSS Styling ---
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

st.markdown("<h1> Breast Cancer Survival Prediction </h1>", unsafe_allow_html=True)

# --- Input Fields ---
st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age", value=st.session_state.get("age", ""), key="age")
    if age.strip():
        if not age.isdigit():
            st.warning(" Age must be a number.")
        elif int(age) < 20:
            st.warning(" Age must be at least 20.")

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
    if lymph_nodes_examined.strip():
        if not lymph_nodes_examined.isdigit():
            st.warning(" Lymph Nodes must be a number.")
        elif int(lymph_nodes_examined) < 0:
            st.warning(" Lymph Nodes must be 0 or more.")

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

# --- Buttons ---
left, right = st.columns(2)
with left:
    if st.button("RESET"):
        st.session_state.clear()
        st.rerun()


with right:
    predict_clicked = st.button("PREDICT")

# --- Prediction logic ---
if predict_clicked:
    required_fields = [st.session_state.get(k, "") for k in field_keys]
    if "" in required_fields:
        st.markdown("""
            <div style='background-color: #fff3cd; padding: 1rem; border-radius: 10px;
                        color: #856404; border: 1px solid #ffeeba;
                        margin-top: 1rem; font-weight: 500;'>
                ⚠️ Please fill in all required fields.
            </div>
        """, unsafe_allow_html=True)
    elif not st.session_state.age.isdigit() or int(st.session_state.age) < 20:
        st.warning(" Age must be a number and at least 20.")
    elif not st.session_state.lymph_nodes_examined.isdigit() or int(st.session_state.lymph_nodes_examined) < 0:
        st.warning(" Lymph Nodes must be a non-negative number.")
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
            <div style='display: flex; justify-content: center; margin-top: 2rem;'>
                <div style='background-color: #ffffff; padding: 2rem; border-radius: 20px;
                            box-shadow: 0 4px 12px rgba(220, 20, 60, 0.15);
                            width: 100%; max-width: 600px; text-align: center;'>
                    <h3 style='color: #c2185b;'> Survival Predictions</h3>
                    <p style='font-size: 22px; font-weight: bold; color: #004d40;'>🩺 5-Year Survival Probability: <span style="color:#004d40;">{survival_5yr:.2f}</span></p>
                    <p style='font-size: 22px; font-weight: bold; color: #004d40;'>🩺 10-Year Survival Probability: <span style="color:#004d40;">{survival_10yr:.2f}</span></p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style='background-color: #d4edda; padding: 1rem; border-radius: 10px;
                        color: #155724; border: 1px solid #c3e6cb;
                        margin-top: 1.5rem; font-weight: 500;'>
                ✅ Patient record successfully saved to MongoDB Atlas.
            </div>
        """, unsafe_allow_html=True)

        patient_data = {key: st.session_state.get(key) for key in field_keys}
        patient_data.update({
            "timestamp": datetime.datetime.now(),
            "survival_5yr": survival_5yr,
            "survival_10yr": survival_10yr
        })
        collection.insert_one(patient_data)

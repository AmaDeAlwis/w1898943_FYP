# Final CoxPH-based Streamlit UI (Reset, Validation, MongoDB Save, Predictions in Container)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pymongo import MongoClient
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from lifelines import CoxPHFitter

# --- Set page config ---
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# --- CSS Styling ---
st.markdown('''
<style>
h1 { color: #ad1457; text-align: center; font-weight: bold; }
.section-title { font-size: 22px; font-weight: bold; color: #ad1457; margin-top: 2rem; margin-bottom: 1rem; }
.result-heading { font-size: 22px; color: #ad1457; font-weight: bold; margin-top: 1rem; margin-bottom: 0.5rem; }
.white-box { background-color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 0 5px rgba(0,0,0,0.1); }
.stButton > button { background-color: #ad1457 !important; color: white !important; border-radius: 10px; font-weight: bold; }
</style>
''', unsafe_allow_html=True)

# --- Load model and scaler ---
cox_model = joblib.load(".streamlit/cox_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- MongoDB ---
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# --- Field Keys ---
field_keys = ["patient_id", "age", "menopausal_status", "tumor_stage", "lymph_nodes_examined", "er_status", "pr_status", "her2_status", "chemotherapy", "surgery", "radiotherapy", "hormone_therapy"]

# --- Reset ---
if "reset_flag" not in st.session_state:
    st.session_state.reset_flag = False
if st.button("RESET"):
    for key in field_keys:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.reset_flag = True
    st.experimental_rerun()

# --- Title ---
st.markdown("<h1>Breast Cancer Survival Prediction</h1>", unsafe_allow_html=True)

# --- Input Fields ---
patient_id = st.text_input("Patient ID (Required)", key="patient_id")
if patient_id:
    prev = list(collection.find({"patient_id": patient_id}))
    if prev:
        with st.expander("Previous Predictions"):
            for r in prev:
                st.write(f"{r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ➔ 5yr: {r['survival_5yr']:.2f}, 10yr: {r['survival_10yr']:.2f}")

st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age", key="age")
    lymph_nodes = st.text_input("Lymph Nodes Examined", key="lymph_nodes_examined")
    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="menopausal_status")
    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="tumor_stage")
with col2:
    er_status = st.selectbox("ER Status", ["", "Positive", "Negative"], key="er_status")
    pr_status = st.selectbox("PR Status", ["", "Positive", "Negative"], key="pr_status")
    her2_status = st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"], key="her2_status")

st.markdown("<p class='section-title'>Treatment Information</p>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    chemotherapy = st.selectbox("Chemotherapy", ["", "Yes", "No"], key="chemotherapy")
    surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"], key="surgery")
with col4:
    radiotherapy = st.selectbox("Radiotherapy", ["", "Yes", "No"], key="radiotherapy")
    hormone_therapy = st.selectbox("Hormone Therapy", ["", "Yes", "No"], key="hormone_therapy")

# --- Predict ---
predict_clicked = st.button("PREDICT")
if predict_clicked:
    required_values = [st.session_state.get(k, "") for k in field_keys]
    if "" in required_values:
        st.error("Please fill all fields before predicting.")
    elif not age.isdigit() or int(age) < 20:
        st.warning("Age must be a number ≥ 20.")
    elif not lymph_nodes.isdigit() or int(lymph_nodes) < 0:
        st.warning("Lymph Nodes must be a non-negative number.")
    else:
        menopausal = 1 if menopausal_status == "Post-menopausal" else 0
        er = 1 if er_status == "Positive" else 0
        pr = 1 if pr_status == "Positive" else 0
        her2_vals = [0, 0, 0, 0]
        her2_opts = ["Gain", "Loss", "Neutral", "Undef"]
        if her2_status in her2_opts:
            her2_vals[her2_opts.index(her2_status)] = 1
        chemo = 1 if chemotherapy == "Yes" else 0
        radio = 1 if radiotherapy == "Yes" else 0
        hormone = 1 if hormone_therapy == "Yes" else 0
        surgery_conserve = 1 if surgery == "Breast-conserving" else 0
        surgery_mastectomy = 1 if surgery == "Mastectomy" else 0

        features = np.array([
            float(age), chemo, er, hormone, menopausal, float(lymph_nodes), pr, radio, int(tumor_stage),
            surgery_conserve, surgery_mastectomy, *her2_vals
        ]).reshape(1, -1)

        features_scaled = scaler.transform(features)
        df_input = pd.DataFrame(features_scaled, columns=cox_model.params_.index)

        surv_func = cox_model.predict_survival_function(df_input)
        times = surv_func.index.values
        surv_5yr = np.interp(60, times, surv_func.values.flatten())
        surv_10yr = np.interp(120, times, surv_func.values.flatten())

        # Save
        collection.insert_one({
            "patient_id": patient_id,
            "timestamp": pd.Timestamp.now(),
            "survival_5yr": float(surv_5yr),
            "survival_10yr": float(surv_10yr)
        })

        st.success("✅ Patient record successfully saved to MongoDB Atlas.")

        st.markdown("<div class='white-box'>", unsafe_allow_html=True)
        st.markdown("<div class='result-heading'>Survival Predictions</div>", unsafe_allow_html=True)
        st.write(f"**5-Year Survival Probability:** {surv_5yr:.2f} ({surv_5yr * 100:.0f}%)")
        st.write(f"**10-Year Survival Probability:** {surv_10yr:.2f} ({surv_10yr * 100:.0f}%)")
        st.markdown("</div>", unsafe_allow_html=True)

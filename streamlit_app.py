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

# Load Cox model and scaler
cox_model = joblib.load(".streamlit/cox_model.pkl")
scaler = joblib.load("scaler.pkl")

# MongoDB connection
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")
st.markdown("""
<style>
h1 { color: #ad1457; text-align: center; font-weight: bold; }
.section-title { font-size: 22px; font-weight: bold; margin-top: 2rem; margin-bottom: 0.5rem; color: #ad1457; }
.stButton button { background-color: #ad1457 !important; color: white !important; border-radius: 10px; font-weight: bold; }
.result-heading { font-size: 22px; color: #c2185b; margin-top: 2rem; font-weight: bold; text-align: left; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1> Breast Cancer Survival Prediction </h1>
<p class='section-title'>Patient Information</p>
""", unsafe_allow_html=True)

# Patient ID
patient_id = st.text_input("Patient ID (Required)")
if patient_id:
    prev = list(collection.find({"patient_id": patient_id}))
    if prev:
        with st.expander("Previous Predictions"):
            for r in prev:
                st.write(f"{r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ➔ 5yr: {r['survival_5yr']:.2f}, 10yr: {r['survival_10yr']:.2f}")

# Clinical Data Section
st.markdown("""
<p class='section-title'>Clinical Information</p>
""", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age")
    lymph_nodes = st.text_input("Lymph Nodes Examined")
    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"])
    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4])
    er_status = st.selectbox("ER Status", ["", "Positive", "Negative"])
    pr_status = st.selectbox("PR Status", ["", "Positive", "Negative"])

with col2:
    her2_status = st.selectbox("HER2 Status", ["", "Gain", "Loss", "Neutral", "Undef"])

# Treatment Information Section
st.markdown("""
<p class='section-title'>Treatment Information</p>
""", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    chemo_status = st.selectbox("Chemotherapy", ["", "Yes", "No"])
    surgery_status = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"])

with col4:
    radio_status = st.selectbox("Radiotherapy", ["", "Yes", "No"])
    hormone_status = st.selectbox("Hormone Therapy", ["", "Yes", "No"])

# Create two columns for Reset and Predict
col5, col6 = st.columns(2)
with col5:
    reset = st.button("RESET")
with col6:
    predict = st.button("PREDICT")

# RESET button clears session state
if reset:
    st.session_state.clear()

# PREDICT button
if predict:
    # Validation
    if "" in [age, lymph_nodes, menopausal_status, tumor_stage, er_status, pr_status, her2_status, chemo_status, radio_status, hormone_status, surgery_status]:
        st.error("❗ Please fill out all fields before predicting.")
    else:
        # Convert categorical inputs to numerical features
        menopausal = 1 if menopausal_status == "Post-menopausal" else 0
        er = 1 if er_status == "Positive" else 0
        pr = 1 if pr_status == "Positive" else 0
        her2_vals = [0, 0, 0, 0]
        her2_opts = ["Gain", "Loss", "Neutral", "Undef"]
        if her2_status in her2_opts:
            her2_vals[her2_opts.index(her2_status)] = 1
        chemo = 1 if chemo_status == "Yes" else 0
        radio = 1 if radio_status == "Yes" else 0
        hormone = 1 if hormone_status == "Yes" else 0
        surgery_conserve = 1 if surgery_status == "Breast-conserving" else 0
        surgery_mastectomy = 1 if surgery_status == "Mastectomy" else 0

        # Assemble feature vector
        features = np.array([
            float(age), chemo, er, hormone, menopausal, float(lymph_nodes), pr, radio, int(tumor_stage),
            surgery_conserve, surgery_mastectomy, *her2_vals
        ]).reshape(1, -1)

        features_scaled = scaler.transform(features)
        df_input = pd.DataFrame(features_scaled, columns=cox_model.params_.index)

        # Predict survival
        surv_func = cox_model.predict_survival_function(df_input)
        times = surv_func.index.values
        surv_5yr = np.interp(60, times, surv_func.values.flatten())
        surv_10yr = np.interp(120, times, surv_func.values.flatten())

        st.success("Prediction complete!")
        st.metric("5-Year Survival Probability", f"{surv_5yr:.2f}")
        st.metric("10-Year Survival Probability", f"{surv_10yr:.2f}")

        # Save to MongoDB
        record = {
            "patient_id": patient_id,
            "timestamp": pd.Timestamp.now(),
            "survival_5yr": float(surv_5yr),
            "survival_10yr": float(surv_10yr)
        }
        collection.insert_one(record)

        # Plot bar chart
        fig, ax = plt.subplots()
        ax.bar(["5-Year", "10-Year"], [surv_5yr, surv_10yr], color="#FF69B4")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # PDF download
        pdf = BytesIO()
        c = canvas.Canvas(pdf, pagesize=letter)
        c.drawString(100, 750, f"Patient ID: {patient_id}")
        c.drawString(100, 730, f"5-Year Survival: {surv_5yr:.2f}")
        c.drawString(100, 710, f"10-Year Survival: {surv_10yr:.2f}")
        c.save()
        pdf.seek(0)

        st.download_button("Download Report", data=pdf, file_name=f"Survival_Report_{patient_id}.pdf", mime="application/pdf")

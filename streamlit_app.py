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

# Set page config and style
st.set_page_config(page_title="Breast Cancer Survival Prediction", layout="wide")
st.markdown("""
<style>
h1 {
    text-align: center;
    color: #ad1457;
}
.section-title {
    font-size: 20px;
    font-weight: bold;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
    color: #ad1457;
}
.stButton > button {
    background-color: #ad1457 !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 10px;
}
.result-heading {
    font-size: 24px;
    font-weight: bold;
    color: #ad1457;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Load model and scaler
cox_model = joblib.load(".streamlit/cox_model.pkl")
scaler = joblib.load("scaler.pkl")

# MongoDB connection
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# Title
st.markdown("<h1>Breast Cancer Survival Prediction</h1>", unsafe_allow_html=True)

# --- FORM START ---
with st.form("patient_form"):
    col_pid1, col_pid2 = st.columns([1, 1])
    with col_pid1:
        patient_id = st.text_input("Patient ID (Required)", key="patient_id")

    st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.text_input("Age", key="age")
        lymph_nodes = st.text_input("Lymph Nodes Examined", key="lymph")
        menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="meno")
        tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="tumor")
    with col2:
        her2 = st.selectbox("HER2 Status", ["", "Gain", "Loss", "Neutral", "Undef"], key="her2")
        er = st.selectbox("ER Status", ["", "Positive", "Negative"], key="er")
        pr = st.selectbox("PR Status", ["", "Positive", "Negative"], key="pr")

    st.markdown("<p class='section-title'>Treatment Information</p>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        chemo = st.selectbox("Chemotherapy", ["", "Yes", "No"], key="chemo")
        surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"], key="surgery")
    with col4:
        radio = st.selectbox("Radiotherapy", ["", "Yes", "No"], key="radio")
        hormone = st.selectbox("Hormone Therapy", ["", "Yes", "No"], key="hormone")

    # Buttons
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        reset = st.form_submit_button("RESET")
    with col_btn2:
        predict = st.form_submit_button("PREDICT")

# RESET logic
if reset:
    for k in st.session_state.keys():
        st.session_state[k] = ""
    st.rerun()

# PREDICT logic
if predict:
    # Validation
    errors = []
    try:
        age_val = float(age)
        if age_val < 20:
            errors.append("Age must be a number and ≥ 20")
    except:
        errors.append("Age must be a number and ≥ 20")
    try:
        lymph_val = float(lymph_nodes)
        if lymph_val < 0:
            errors.append("Lymph Nodes Examined must be a non-negative number")
    except:
        errors.append("Lymph Nodes Examined must be a non-negative number")

    if any(x == "" for x in [patient_id, menopausal_status, er, pr, her2, chemo, radio, hormone, surgery, tumor_stage]):
        errors.append("Please fill all fields.")

    for e in errors:
        st.warning(e)

    if not errors:
        menopausal = 1 if menopausal_status == "Post-menopausal" else 0
        er_val = 1 if er == "Positive" else 0
        pr_val = 1 if pr == "Positive" else 0
        her2_vals = [0, 0, 0, 0]
        her2_opts = ["Gain", "Loss", "Neutral", "Undef"]
        if her2 in her2_opts:
            her2_vals[her2_opts.index(her2)] = 1
        chemo_val = 1 if chemo == "Yes" else 0
        radio_val = 1 if radio == "Yes" else 0
        hormone_val = 1 if hormone == "Yes" else 0
        surgery_conserve = 1 if surgery == "Breast-conserving" else 0
        surgery_mastectomy = 1 if surgery == "Mastectomy" else 0

        features = np.array([
            age_val, chemo_val, er_val, hormone_val, menopausal,
            lymph_val, pr_val, radio_val, int(tumor_stage),
            surgery_conserve, surgery_mastectomy, *her2_vals
        ]).reshape(1, -1)

        features_scaled = scaler.transform(features)
        df_input = pd.DataFrame(features_scaled, columns=cox_model.params_.index)

        surv_func = cox_model.predict_survival_function(df_input)
        times = surv_func.index.values
        surv_5yr = np.interp(60, times, surv_func.values.flatten())
        surv_10yr = np.interp(120, times, surv_func.values.flatten())

        st.success("Patient record successfully saved!")

        st.markdown("""
        <div style="background-color:white;padding:20px;border-radius:10px">
        <p class='result-heading'>Survival Predictions</p>
        <p><b>5-Year Survival Probability:</b> {:.2f} ({:.0f}%)</p>
        <p><b>10-Year Survival Probability:</b> {:.2f} ({:.0f}%)</p>
        </div>
        """.format(surv_5yr, surv_5yr*100, surv_10yr, surv_10yr*100), unsafe_allow_html=True)

        # Save record
        record = {
            "patient_id": patient_id,
            "timestamp": pd.Timestamp.now(),
            "survival_5yr": float(surv_5yr),
            "survival_10yr": float(surv_10yr)
        }
        collection.insert_one(record)

        st.markdown("<p class='result-heading'>Results Overview</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])

        # Bar chart
        with col1:
            fig, ax = plt.subplots()
            ax.bar(["5-Year", "10-Year"], [surv_5yr, surv_10yr], color="#FF69B4")
            for i, val in enumerate([surv_5yr, surv_10yr]):
                ax.text(i, val + 0.02, f"{val:.2f}", ha='center', fontsize=10)
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        # Recommendation
        with col2:
            if surv_5yr < 0.5:
                st.error("Low Survival Chance")
                st.info("Patient shows low probability. Consider aggressive treatment planning.")
            else:
                st.success("High Survival Chance")
                st.info("Patient has a favorable survival outlook. Continue regular monitoring.")

        # Survival curve
        with col3:
            fig2, ax2 = plt.subplots()
            ax2.plot(surv_func.index, surv_func.values.flatten(), color="#d63384")
            ax2.set_title("Survival Curve")
            ax2.set_xlabel("Time (Months)")
            ax2.set_ylabel("Survival Probability")
            st.pyplot(fig2)

        # PDF download
        pdf = BytesIO()
        c = canvas.Canvas(pdf, pagesize=letter)
        c.drawString(100, 750, f"Patient ID: {patient_id}")
        c.drawString(100, 730, f"5-Year Survival: {surv_5yr:.2f}")
        c.drawString(100, 710, f"10-Year Survival: {surv_10yr:.2f}")
        c.save()
        pdf.seek(0)

        st.download_button("Download Report", data=pdf, file_name=f"Survival_Report_{patient_id}.pdf", mime="application/pdf")

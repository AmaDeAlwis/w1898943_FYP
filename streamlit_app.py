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

# Load models and scaler
cox_model = joblib.load(".streamlit/cox_model.pkl")
scaler = joblib.load("scaler.pkl")

# MongoDB connection
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

st.set_page_config(page_title="Breast Cancer Survival Prediction", layout="wide")
st.markdown("""
    <style>
    h1, .section-title, .result-title {
        color: #ad1457;
        font-weight: bold;
    }
    .stButton button {
        background-color: #ad1457 !important;
        color: white !important;
        border-radius: 10px !important;
    }
    .white-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Breast Cancer Survival Prediction</h1>", unsafe_allow_html=True)

# Patient ID
patient_id = st.text_input("Patient ID (Required)", key="patient_id")
if patient_id:
    prev_records = list(collection.find({"patient_id": patient_id}))
    if prev_records:
        with st.expander("Previous Predictions"):
            for r in prev_records:
                st.write(f"{r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ➜ 5yr: {r['survival_5yr']:.2f}, 10yr: {r['survival_10yr']:.2f}")

# Clinical Information
st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    age = st.text_input("Age", key="age")
    if age:
        try:
            if int(age) < 20:
                st.warning("Age must be a number and ≥ 20")
        except:
            st.warning("Age must be a number and ≥ 20")

    lymph_nodes = st.text_input("Lymph Nodes Examined", key="lymph_nodes")
    if lymph_nodes:
        try:
            if int(lymph_nodes) < 0:
                st.warning("Lymph Nodes Examined must be a non-negative number")
        except:
            st.warning("Lymph Nodes Examined must be a non-negative number")

    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="meno")
    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="tumor")

with col2:
    her2 = st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"], key="her2")
    er = st.selectbox("ER Status", ["", "Positive", "Negative"], key="er")
    pr = st.selectbox("PR Status", ["", "Positive", "Negative"], key="pr")

# Treatment Information
st.markdown("<p class='section-title'>Treatment Information</p>", unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    chemo = st.selectbox("Chemotherapy", ["", "Yes", "No"], key="chemo")
    surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"], key="surgery")

with col4:
    radio = st.selectbox("Radiotherapy", ["", "Yes", "No"], key="radio")
    hormone = st.selectbox("Hormone Therapy", ["", "Yes", "No"], key="hormone")

# Buttons
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("RESET"):
        for k in st.session_state.keys():
            st.session_state[k] = ""
        st.experimental_rerun()

with col_btn2:
    predict = st.button("PREDICT")

# Prediction
if predict:
    # Ensure all fields filled
    required_fields = [age, lymph_nodes, menopausal_status, tumor_stage, her2, er, pr, chemo, radio, hormone, surgery]
    if "" in required_fields:
        st.error("Please fill all required fields before predicting.")
    else:
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
            float(age), chemo_val, er_val, hormone_val, menopausal, float(lymph_nodes), pr_val, radio_val, int(tumor_stage),
            surgery_conserve, surgery_mastectomy, *her2_vals
        ]).reshape(1, -1)

        features_scaled = scaler.transform(features)
        df_input = pd.DataFrame(features_scaled, columns=cox_model.params_.index)

        surv_func = cox_model.predict_survival_function(df_input)
        times = surv_func.index.values
        surv_5yr = np.interp(60, times, surv_func.values.flatten())
        surv_10yr = np.interp(120, times, surv_func.values.flatten())

        # Save to MongoDB
        record = {
            "patient_id": patient_id,
            "timestamp": pd.Timestamp.now(),
            "survival_5yr": float(surv_5yr),
            "survival_10yr": float(surv_10yr)
        }
        collection.insert_one(record)

        # Display Survival Predictions
        st.markdown("""
            <div class='white-container'>
            <h3 class='section-title'>Survival Predictions</h3>
            <p><b>5-Year Survival Probability:</b> {:.2f} ({:.0f}%)</p>
            <p><b>10-Year Survival Probability:</b> {:.2f} ({:.0f}%)</p>
            </div>
        """.format(surv_5yr, surv_5yr*100, surv_10yr, surv_10yr*100), unsafe_allow_html=True)

        st.success("Patient record successfully saved!")

        # Results Overview
        st.markdown("<h3 class='section-title'>Results Overview</h3>", unsafe_allow_html=True)
        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            fig, ax = plt.subplots()
            ax.bar(["5-Year", "10-Year"], [surv_5yr, surv_10yr], color="#FF69B4")
            for i, v in enumerate([surv_5yr, surv_10yr]):
                ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        with col_res2:
            if surv_5yr > 0.7 and surv_10yr > 0.7:
                st.success("High Survival Chance")
                st.info("Patient has a favorable survival outlook. Continue regular monitoring.")
            else:
                st.error("Low Survival Chance")
                st.warning("Patient shows low probability. Consider aggressive treatment planning.")

        with col_res3:
            fig2, ax2 = plt.subplots()
            ax2.plot(times, surv_func.values.flatten(), color="#ad1457")
            ax2.set_xlabel("Time (Months)")
            ax2.set_ylabel("Survival Probability")
            ax2.set_title("Survival Curve")
            st.pyplot(fig2)

        # PDF download
        pdf = BytesIO()
        c = canvas.Canvas(pdf, pagesize=letter)
        c.drawString(100, 750, f"Patient ID: {patient_id}")
        c.drawString(100, 730, f"5-Year Survival: {surv_5yr:.2f} ({surv_5yr*100:.0f}%)")
        c.drawString(100, 710, f"10-Year Survival: {surv_10yr:.2f} ({surv_10yr*100:.0f}%)")
        c.save()
        pdf.seek(0)

        st.download_button("Download Report", data=pdf, file_name=f"Survival_Report_{patient_id}.pdf", mime="application/pdf")

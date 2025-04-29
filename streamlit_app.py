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

# Load model and scaler
cox_model = joblib.load(".streamlit/cox_model.pkl")
scaler = joblib.load("scaler.pkl")

# MongoDB
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# Page configuration
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")
st.markdown("""
<style>
h1, .main-title { color: #ad1457; font-weight: bold; }
.section-title { font-size: 20px; font-weight: bold; margin-top: 2rem; margin-bottom: 0.5rem; color: #ad1457; }
.stButton button { background-color: #ad1457 !important; color: white !important; border-radius: 10px; font-weight: bold; }
.metric-box { background-color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
.result-heading { font-size: 24px; font-weight: bold; color: #ad1457; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Breast Cancer Survival Prediction</h1>", unsafe_allow_html=True)

# Session state reset handler
if "reset_flag" not in st.session_state:
    st.session_state.reset_flag = False

# Form Section
with st.form(key="survival_form"):
    patient_id = st.text_input("Patient ID (Required)")

    if patient_id:
        prev = list(collection.find({"patient_id": patient_id}))
        if prev:
            with st.expander("Previous Predictions"):
                for r in prev:
                    st.write(f"{r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ➜ 5yr: {r['survival_5yr']:.2f}, 10yr: {r['survival_10yr']:.2f}")

    st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.text_input("Age")
        if age and (not age.isnumeric() or int(age) < 20):
            st.warning("Age must be a number and ≥ 20")
        lymph_nodes = st.text_input("Lymph Nodes Examined")
        if lymph_nodes and (not lymph_nodes.isnumeric() or int(lymph_nodes) < 0):
            st.warning("Lymph Nodes Examined must be a non-negative number")
        menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"])
        tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4])
    with col2:
        her2 = st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"])
        er = st.selectbox("ER Status", ["", "Positive", "Negative"])
        pr = st.selectbox("PR Status", ["", "Positive", "Negative"])

    st.markdown("<p class='section-title'>Treatment Information</p>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        chemo = st.selectbox("Chemotherapy", ["", "Yes", "No"])
        surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"])
    with col4:
        radio = st.selectbox("Radiotherapy", ["", "Yes", "No"])
        hormone = st.selectbox("Hormone Therapy", ["", "Yes", "No"])

    col5, col6 = st.columns(2)
    with col5:
        reset = st.form_submit_button("RESET")
    with col6:
        predict = st.form_submit_button("PREDICT")

if reset:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

if predict:
    if "" in [age, lymph_nodes, menopausal_status, tumor_stage, er, pr, her2, chemo, radio, hormone, surgery]:
        st.error("❗ Please fill out all fields before predicting.")
    else:
        menopausal = 1 if menopausal_status == "Post-menopausal" else 0
        er = 1 if er == "Positive" else 0
        pr = 1 if pr == "Positive" else 0
        her2_vals = [0, 0, 0, 0]
        her2_opts = ["Gain", "Loss", "Neutral", "Undef"]
        if her2 in her2_opts:
            her2_vals[her2_opts.index(her2)] = 1
        chemo = 1 if chemo == "Yes" else 0
        radio = 1 if radio == "Yes" else 0
        hormone = 1 if hormone == "Yes" else 0
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

        st.markdown("""
        <div class='metric-box'>
            <h3 style='color:#ad1457;'>Survival Predictions</h3>
            <p>5-Year Survival Probability: <b>{:.2f}</b> ({:.0f}%)</p>
            <p>10-Year Survival Probability: <b>{:.2f}</b> ({:.0f}%)</p>
        </div>
        """.format(surv_5yr, surv_5yr*100, surv_10yr, surv_10yr*100), unsafe_allow_html=True)

        st.success("Patient record successfully saved!")
        collection.insert_one({
            "patient_id": patient_id,
            "timestamp": pd.Timestamp.now(),
            "survival_5yr": float(surv_5yr),
            "survival_10yr": float(surv_10yr)
        })

        colA, colB, colC = st.columns(3)
        with colA:
            fig, ax = plt.subplots()
            ax.bar(["5-Year", "10-Year"], [surv_5yr, surv_10yr], color="#FF69B4")
            for i, v in enumerate([surv_5yr, surv_10yr]):
                ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        with colB:
            st.markdown("<h3 class='result-heading'>Results Overview</h3>", unsafe_allow_html=True)
            if surv_5yr < 0.5:
                st.error("Low Survival Chance")
                st.info("Patient shows low probability. Consider aggressive treatment planning.")
            else:
                st.success("High Survival Chance")
                st.info("Patient has a favorable survival outlook. Continue regular monitoring.")

        with colC:
            fig2, ax2 = plt.subplots()
            ax2.plot(surv_func.index, surv_func.values.flatten(), color="#c2185b")
            ax2.set_title("Survival Curve")
            ax2.set_xlabel("Time (Months)")
            ax2.set_ylabel("Survival Probability")
            st.pyplot(fig2)

        pdf = BytesIO()
        c = canvas.Canvas(pdf, pagesize=letter)
        c.drawString(100, 750, f"Patient ID: {patient_id}")
        c.drawString(100, 730, f"5-Year Survival: {surv_5yr:.2f}")
        c.drawString(100, 710, f"10-Year Survival: {surv_10yr:.2f}")
        c.save()
        pdf.seek(0)

        st.download_button("Download Report", data=pdf, file_name=f"Survival_Report_{patient_id}.pdf", mime="application/pdf")

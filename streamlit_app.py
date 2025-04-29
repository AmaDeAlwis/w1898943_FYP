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
.result-container { background-color: white; padding: 1.5rem; border-radius: 10px; margin-top: 1rem; }
.result-heading { font-size: 22px; color: #c2185b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1> Breast Cancer Survival Prediction </h1>", unsafe_allow_html=True)

# Patient ID
st.markdown("<p class='section-title'>Patient Information</p>", unsafe_allow_html=True)
patient_id = st.text_input("Patient ID (Required)")
if patient_id:
    prev = list(collection.find({"patient_id": patient_id}))
    if prev:
        with st.expander("Previous Predictions"):
            for r in prev:
                st.write(f"{r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ➜ 5yr: {r['survival_5yr']:.2f}, 10yr: {r['survival_10yr']:.2f}")

# Input fields
st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age")
    lymph_nodes = st.text_input("Lymph Nodes Examined")
    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"])
    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4])
with col2:
    her2 = st.selectbox("HER2 Status", ["", "Gain", "Loss", "Neutral", "Undef"])
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

# Buttons
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    reset = st.button("RESET")
with col_btn2:
    predict = st.button("PREDICT")

if reset:
    st.session_state.clear()

if predict:
    # Validation
    if not age.isdigit() or int(age) < 20:
        st.warning("⚠️ Age must be a number and ≥ 20")
    elif not lymph_nodes.isdigit() or int(lymph_nodes) < 0:
        st.warning("⚠️ Lymph Nodes Examined must be a non-negative number")
    elif "" in [menopausal_status, er, pr, her2, chemo, radio, hormone, surgery, tumor_stage]:
        st.error("❗ Please fill out all fields before predicting.")
    else:
        age = float(age)
        lymph_nodes = float(lymph_nodes)
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
            age, chemo, er, hormone, menopausal, lymph_nodes, pr, radio, int(tumor_stage),
            surgery_conserve, surgery_mastectomy, *her2_vals
        ]).reshape(1, -1)

        features_scaled = scaler.transform(features)
        df_input = pd.DataFrame(features_scaled, columns=cox_model.params_.index)

        surv_func = cox_model.predict_survival_function(df_input)
        times = surv_func.index.values
        surv_5yr = np.interp(60, times, surv_func.values.flatten())
        surv_10yr = np.interp(120, times, surv_func.values.flatten())

        # Output container
        with st.container():
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            st.markdown("<p class='result-heading'>Survival Predictions</p>", unsafe_allow_html=True)
            st.markdown(f"5-Year Survival Probability: <b>{surv_5yr:.2f}</b>", unsafe_allow_html=True)
            st.markdown(f"10-Year Survival Probability: <b>{surv_10yr:.2f}</b>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.success("✅ Patient record successfully saved!")

        record = {
            "patient_id": patient_id,
            "timestamp": pd.Timestamp.now(),
            "survival_5yr": float(surv_5yr),
            "survival_10yr": float(surv_10yr)
        }
        collection.insert_one(record)

        # Columns for visualization and interpretation
        col_chart, col_text, col_curve = st.columns(3)

        with col_chart:
            fig, ax = plt.subplots()
            ax.bar(["5-Year", "10-Year"], [surv_5yr, surv_10yr], color="#FF69B4")
            ax.set_ylim(0, 1)
            for i, val in enumerate([surv_5yr, surv_10yr]):
                ax.text(i, val + 0.02, f"{val:.2f}", ha='center')
            st.pyplot(fig)

        with col_text:
            st.subheader("Results Overview")
            if surv_5yr < 0.5 or surv_10yr < 0.5:
                st.error("Low Survival Chance")
                st.info("Patient shows low probability. Consider aggressive treatment planning.")
            else:
                st.success("High Survival Chance")
                st.info("Patient has a favorable survival outlook. Continue regular monitoring.")

        with col_curve:
            fig, ax = plt.subplots()
            ax.plot(times, surv_func.values.flatten(), label="Survival Curve", color="#ad1457")
            ax.set_xlabel("Time (Months)")
            ax.set_ylabel("Survival Probability")
            ax.set_title("Survival Curve")
            ax.grid(True)
            st.pyplot(fig)

        pdf = BytesIO()
        c = canvas.Canvas(pdf, pagesize=letter)
        c.drawString(100, 750, f"Patient ID: {patient_id}")
        c.drawString(100, 730, f"5-Year Survival: {surv_5yr:.2f}")
        c.drawString(100, 710, f"10-Year Survival: {surv_10yr:.2f}")
        c.save()
        pdf.seek(0)

        st.download_button("Download Report", data=pdf, file_name=f"Survival_Report_{patient_id}.pdf", mime="application/pdf")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from lifelines import CoxPHFitter
import joblib

# Load model and scaler
cox_model = joblib.load(".streamlit/cox_model.pkl")
scaler = joblib.load("scaler.pkl")

# MongoDB Connection
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# Page config
st.set_page_config(page_title="Breast Cancer Survival Prediction", layout="wide")

# Custom CSS
st.markdown("""
<style>
    h1 {
        text-align: center;
        color: #ad1457;
        font-weight: bold;
    }
    .section-title {
        font-size: 22px;
        font-weight: bold;
        margin-top: 2rem;
        color: #ad1457;
    }
    .stButton button {
        background-color: #ad1457 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Breast Cancer Survival Prediction</h1>", unsafe_allow_html=True)

# RESET button
if st.button("RESET"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# --- Patient ID ---
patient_id = st.text_input("Patient ID (Required)", value=st.session_state.get("patient_id", ""))

# Fetch previous predictions
if patient_id:
    prev = list(collection.find({"patient_id": patient_id}))
    if prev:
        with st.expander("Previous Predictions"):
            for r in prev:
                st.write(f"{r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ➔ 5yr: {r['survival_5yr']:.2f}, 10yr: {r['survival_10yr']:.2f}")

# --- Input Section ---
st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    age = st.text_input("Age", value=st.session_state.get("age", ""))
    lymph_nodes = st.text_input("Lymph Nodes Examined", value=st.session_state.get("lymph_nodes", ""))
    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"])
    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4])

    if age and (not age.isnumeric() or int(age) < 20):
        st.warning("Age must be a number and ≥ 20")

    if lymph_nodes and (not lymph_nodes.lstrip('-').isnumeric() or int(lymph_nodes) < 0):
        st.warning("Lymph Nodes Examined must be a non-negative number")

with col2:
    her2 = st.selectbox("HER2 Status", ["", "Gain", "Loss", "Neutral", "Undef"])
    er = st.selectbox("ER Status", ["", "Positive", "Negative"])
    pr = st.selectbox("PR Status", ["", "Positive", "Negative"])

# --- Treatment Section ---
st.markdown("<p class='section-title'>Treatment Information</p>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    chemo = st.selectbox("Chemotherapy", ["", "Yes", "No"])
    surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"])
with col4:
    radio = st.selectbox("Radiotherapy", ["", "Yes", "No"])
    hormone = st.selectbox("Hormone Therapy", ["", "Yes", "No"])

# --- Predict Button ---
if st.button("PREDICT"):
    required_fields = [age, lymph_nodes, menopausal_status, tumor_stage, her2, er, pr, chemo, radio, hormone, surgery]
    if "" in required_fields or not age.isnumeric() or not lymph_nodes.isnumeric():
        st.error("Please fill out all fields correctly before predicting.")
    else:
        # Save input values to session state
        st.session_state.update({
            "age": age,
            "lymph_nodes": lymph_nodes
        })

        menopausal = 1 if menopausal_status == "Post-menopausal" else 0
        er = 1 if er == "Positive" else 0
        pr = 1 if pr == "Positive" else 0
        chemo = 1 if chemo == "Yes" else 0
        radio = 1 if radio == "Yes" else 0
        hormone = 1 if hormone == "Yes" else 0
        surgery_conserve = 1 if surgery == "Breast-conserving" else 0
        surgery_mastectomy = 1 if surgery == "Mastectomy" else 0

        her2_vals = [0, 0, 0, 0]
        her2_opts = ["Gain", "Loss", "Neutral", "Undef"]
        if her2 in her2_opts:
            her2_vals[her2_opts.index(her2)] = 1

        features = np.array([
            int(age), chemo, er, hormone, menopausal, int(lymph_nodes), pr, radio, int(tumor_stage),
            surgery_conserve, surgery_mastectomy, *her2_vals
        ]).reshape(1, -1)

        features_scaled = scaler.transform(features)
        df_input = pd.DataFrame(features_scaled, columns=cox_model.params_.index)

        surv_func = cox_model.predict_survival_function(df_input)
        times = surv_func.index.values
        surv_5yr = np.interp(60, times, surv_func.values.flatten())
        surv_10yr = np.interp(120, times, surv_func.values.flatten())

        st.session_state["surv_5yr"] = surv_5yr
        st.session_state["surv_10yr"] = surv_10yr
        st.session_state["predicted"] = True

        # Save to MongoDB
        collection.insert_one({
            "patient_id": patient_id,
            "timestamp": pd.Timestamp.now(),
            "survival_5yr": float(surv_5yr),
            "survival_10yr": float(surv_10yr)
        })

# --- Output Section ---
if st.session_state.get("predicted"):
    st.success("Patient record successfully saved!")

    # White container
    with st.container():
        st.markdown("<h3 style='color:#ad1457'>Survival Predictions</h3>", unsafe_allow_html=True)
        st.write(f"**5-Year Survival Probability:** {st.session_state['surv_5yr']:.2f} ({int(st.session_state['surv_5yr']*100)}%)")
        st.write(f"**10-Year Survival Probability:** {st.session_state['surv_10yr']:.2f} ({int(st.session_state['surv_10yr']*100)}%)")

    # Results Overview
    st.markdown("<h3 style='color:#ad1457'>Results Overview</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.2, 1])

    with c1:
        fig, ax = plt.subplots()
        ax.bar(["5-Year", "10-Year"], [st.session_state['surv_5yr'], st.session_state['surv_10yr']], color="#FF69B4")
        ax.set_ylim(0, 1)
        for i, v in enumerate([st.session_state['surv_5yr'], st.session_state['surv_10yr']]):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center', weight='bold')
        st.pyplot(fig)

    with c2:
        risk_tag = "High" if st.session_state['surv_5yr'] < 0.5 else "Low" if st.session_state['surv_5yr'] > 0.8 else "Moderate"
        st.markdown(f"<div style='background-color:#f0f0f0;padding:10px;border-radius:10px;'>"
                    f"<b>{risk_tag} Survival Chance</b></div>", unsafe_allow_html=True)
        msg = {
            "High": "Patient has a favorable survival outlook. Continue regular monitoring.",
            "Moderate": "Patient has moderate risk. Re-evaluation and frequent monitoring recommended.",
            "Low": "Patient shows low probability. Consider aggressive treatment planning."
        }
        st.info(msg[risk_tag])

    with c3:
        fig2, ax2 = plt.subplots()
        surv_func = cox_model.predict_survival_function(df_input)
        ax2.plot(surv_func.index, surv_func.values.flatten(), color="#ad1457")
        ax2.set_title("Survival Curve")
        ax2.set_xlabel("Time (Months)")
        ax2.set_ylabel("Survival Probability")
        st.pyplot(fig2)

    # Download button (PDF)
    pdf = BytesIO()
    c = canvas.Canvas(pdf, pagesize=letter)
    c.drawString(100, 750, f"Patient ID: {patient_id}")
    c.drawString(100, 730, f"5-Year Survival: {st.session_state['surv_5yr']:.2f}")
    c.drawString(100, 710, f"10-Year Survival: {st.session_state['surv_10yr']:.2f}")
    c.save()
    pdf.seek(0)
    st.download_button("Download Report", data=pdf, file_name=f"Survival_Report_{patient_id}.pdf", mime="application/pdf")

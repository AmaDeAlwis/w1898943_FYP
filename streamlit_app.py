import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
import joblib
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from lifelines import CoxPHFitter

# Load model and scaler
cox_model = joblib.load(".streamlit/cox_model.pkl")
scaler = joblib.load("scaler.pkl")

# MongoDB setup
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# Page setup
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")
st.markdown("""
<style>
h1 { color: #ad1457; text-align: center; font-weight: bold; }
.section-title { font-size: 22px; font-weight: bold; margin-top: 2rem; margin-bottom: 0.5rem; color: #ad1457; }
.result-heading { font-size: 22px; color: #ad1457; font-weight: bold; text-align: left; }
.metric-container {
    background-color: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.stButton button {
    background-color: #ad1457 !important;
    color: white !important;
    font-weight: bold;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Breast Cancer Survival Prediction</h1>", unsafe_allow_html=True)

# Patient ID
patient_id = st.text_input("Patient ID (Required)", key="patient_id")
if patient_id:
    prev = list(collection.find({"patient_id": patient_id}))
    if prev:
        with st.expander("Previous Predictions"):
            for r in prev:
                st.write(f"{r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ➜ 5yr: {r['survival_5yr']:.2f}, 10yr: {r['survival_10yr']:.2f}")

# Input Fields
st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age", key="age")
    lymph_nodes = st.text_input("Lymph Nodes Examined", key="lymph")
    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="meno")
    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="stage")
with col2:
    her2 = st.selectbox("HER2 Status", ["", "Gain", "Loss", "Neutral", "Undef"], key="her2")
    er = st.selectbox("ER Status", ["", "Positive", "Negative"], key="er")
    pr = st.selectbox("PR Status", ["", "Positive", "Negative"], key="pr")

# Instant validation messages
if age and (not age.isdigit() or int(age) < 20):
    st.warning("Age must be a number and ≥ 20")
if lymph_nodes:
    try:
        if int(lymph_nodes) < 0:
            st.warning("Lymph Nodes Examined must be a non-negative number")
    except ValueError:
        st.warning("Lymph Nodes Examined must be a number")

# Treatment
st.markdown("<p class='section-title'>Treatment Information</p>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    chemo = st.selectbox("Chemotherapy", ["", "Yes", "No"], key="chemo")
    surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"], key="surgery")
with col4:
    radio = st.selectbox("Radiotherapy", ["", "Yes", "No"], key="radio")
    hormone = st.selectbox("Hormone Therapy", ["", "Yes", "No"], key="hormone")

# Action Buttons
col5, col6 = st.columns(2)
with col5:
    if st.button("RESET"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()
with col6:
    if st.button("PREDICT"):
        # Validation
        if "" in [age, lymph_nodes, menopausal_status, tumor_stage, er, pr, her2, chemo, radio, hormone, surgery]:
            st.error("❗ Please fill out all fields before predicting.")
        else:
            try:
                age_val = float(age)
                lymph_val = float(lymph_nodes)
                if age_val < 20:
                    st.error("Age must be ≥ 20")
                elif lymph_val < 0:
                    st.error("Lymph Nodes Examined must be non-negative")
                else:
                    # Encode
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
                        age_val, chemo_val, er_val, hormone_val, menopausal, lymph_val, pr_val, radio_val, int(tumor_stage),
                        surgery_conserve, surgery_mastectomy, *her2_vals
                    ]).reshape(1, -1)

                    features_scaled = scaler.transform(features)
                    df_input = pd.DataFrame(features_scaled, columns=cox_model.params_.index)

                    surv_func = cox_model.predict_survival_function(df_input)
                    times = surv_func.index.values
                    surv_5yr = np.interp(60, times, surv_func.values.flatten())
                    surv_10yr = np.interp(120, times, surv_func.values.flatten())

                    with st.container():
                        st.markdown("""
                        <div class='metric-container'>
                        <p class='section-title'>Survival Predictions</p>
                        <p>5-Year Survival Probability: <b>{:.2f}</b></p>
                        <p>10-Year Survival Probability: <b>{:.2f}</b></p>
                        </div>
                        """.format(surv_5yr, surv_10yr), unsafe_allow_html=True)

                    # Save to MongoDB
                    record = {
                        "patient_id": patient_id,
                        "timestamp": pd.Timestamp.now(),
                        "survival_5yr": float(surv_5yr),
                        "survival_10yr": float(surv_10yr)
                    }
                    collection.insert_one(record)
                    st.success("✅ Patient record successfully saved!")

                    # Plot charts and summary
                    col_a, col_b, col_c = st.columns([1, 1.2, 1])
                    with col_a:
                        fig, ax = plt.subplots()
                        ax.bar(["5-Year", "10-Year"], [surv_5yr, surv_10yr], color="#FF69B4")
                        ax.set_ylim(0, 1)
                        for i, v in enumerate([surv_5yr, surv_10yr]):
                            ax.text(i, v + 0.02, f"{v:.2f}", ha='center', color='black')
                        st.pyplot(fig)

                    with col_b:
                        st.markdown("<p class='result-heading'>Results Overview</p>", unsafe_allow_html=True)
                        if surv_5yr < 0.4 or surv_10yr < 0.4:
                            st.error("Low Survival Chance")
                            st.info("Patient shows low probability. Consider aggressive treatment planning.")
                        elif surv_5yr > 0.8 and surv_10yr > 0.8:
                            st.success("High Survival Chance")
                            st.info("Patient has a favorable survival outlook. Continue regular monitoring.")
                        else:
                            st.warning("Moderate Survival Chance")
                            st.info("Requires balanced approach. Consult with medical team.")

                    with col_c:
                        fig2, ax2 = plt.subplots()
                        ax2.plot(times, surv_func.values.flatten(), color='#C2185B')
                        ax2.set_title("Survival Curve")
                        ax2.set_xlabel("Time (Months)")
                        ax2.set_ylabel("Survival Probability")
                        st.pyplot(fig2)

                    # PDF Report
                    pdf = BytesIO()
                    c = canvas.Canvas(pdf, pagesize=letter)
                    c.drawString(100, 750, f"Patient ID: {patient_id}")
                    c.drawString(100, 730, f"5-Year Survival: {surv_5yr:.2f}")
                    c.drawString(100, 710, f"10-Year Survival: {surv_10yr:.2f}")
                    c.save()
                    pdf.seek(0)

                    st.download_button("Download Report", data=pdf, file_name=f"Survival_Report_{patient_id}.pdf", mime="application/pdf")

            except ValueError:
                st.error("Invalid numeric input.")

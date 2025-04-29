import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from lifelines import CoxPHFitter
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import joblib

# Load model and scaler
cox_model = joblib.load(".streamlit/cox_model.pkl")
scaler = joblib.load("scaler.pkl")

# MongoDB connection
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# Styling
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")
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
.stButton button {
    background-color: #ad1457;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 8px 16px;
}
.metric-label {
    font-size: 18px;
    color: #ad1457;
    font-weight: bold;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Breast Cancer Survival Prediction</h1>", unsafe_allow_html=True)

# Initialize session
if 'predicted' not in st.session_state:
    st.session_state.predicted = False

# Form Inputs
with st.form("prediction_form"):
    patient_id = st.text_input("Patient ID (Required)")

    st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.text_input("Age", key="age")
        lymph_nodes = st.text_input("Lymph Nodes Examined", key="nodes")
        menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="meno")
        tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="stage")
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

    submitted = st.form_submit_button("PREDICT")
    reset = st.form_submit_button("RESET")

# Validation on input change
if age and (not age.isdigit() or int(age) < 20):
    st.warning("Age must be a number and ≥ 20", icon="⚠️")
if lymph_nodes and (not lymph_nodes.isdigit() or int(lymph_nodes) < 0):
    st.warning("Lymph Nodes Examined must be a non-negative number", icon="⚠️")

# Reset functionality
if reset:
    for key in list(st.session_state.keys()):
        if key not in ["_session_state", "predicted"]:
            del st.session_state[key]
    st.session_state.predicted = False
    st.experimental_rerun()

# Prediction process
if submitted:
    required_fields = [age, lymph_nodes, menopausal_status, tumor_stage, her2, er, pr, chemo, radio, hormone, surgery, patient_id]
    if "" in required_fields or not age.isdigit() or int(age) < 20 or not lymph_nodes.isdigit() or int(lymph_nodes) < 0:
        st.error("Please fill all required fields correctly before predicting.")
    else:
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
            float(age), chemo, er, hormone, menopausal, float(lymph_nodes), pr, radio, int(tumor_stage),
            surgery_conserve, surgery_mastectomy, *her2_vals
        ]).reshape(1, -1)

        features_scaled = scaler.transform(features)
        df_input = pd.DataFrame(features_scaled, columns=cox_model.params_.index)

        surv_func = cox_model.predict_survival_function(df_input)
        times = surv_func.index.values
        surv_5yr = np.interp(60, times, surv_func.values.flatten())
        surv_10yr = np.interp(120, times, surv_func.values.flatten())

        # Save to DB
        record = {
            "patient_id": patient_id,
            "timestamp": pd.Timestamp.now(),
            "survival_5yr": float(surv_5yr),
            "survival_10yr": float(surv_10yr)
        }
        collection.insert_one(record)
        st.success("Patient record successfully saved!")

        st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px;'>
            <h3 style='color: #ad1457;'>Survival Predictions</h3>
            <p><strong>5-Year Survival Probability:</strong> {0:.2f} ({1:.0f}%)</p>
            <p><strong>10-Year Survival Probability:</strong> {2:.2f} ({3:.0f}%)</p>
        </div>
        """.format(surv_5yr, surv_5yr * 100, surv_10yr, surv_10yr * 100), unsafe_allow_html=True)

        st.markdown("<h3 class='section-title'>Results Overview</h3>", unsafe_allow_html=True)
        col_res1, col_res2, col_res3 = st.columns([1.5, 1.5, 1.5])

        # Bar chart
        with col_res1:
            fig, ax = plt.subplots()
            ax.bar(["5-Year", "10-Year"], [surv_5yr, surv_10yr], color="#FF69B4")
            for i, v in enumerate([surv_5yr, surv_10yr]):
                ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        # Risk tags
        with col_res2:
            if surv_5yr < 0.5 or surv_10yr < 0.5:
                st.error("Low Survival Chance")
                st.info("Patient shows low probability. Consider aggressive treatment planning.")
            else:
                st.success("High Survival Chance")
                st.info("Patient has a favorable survival outlook. Continue regular monitoring.")

        # Survival curve
        with col_res3:
            fig2, ax2 = plt.subplots()
            ax2.plot(surv_func.index, surv_func.values.flatten(), color='deeppink')
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

        st.session_state.predicted = True

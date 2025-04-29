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

# --- Reset mechanism ---
if st.query_params.get("reset") == "1":
    st.query_params.clear()
    st.session_state.clear()
    st.experimental_rerun()

# --- Load model and scaler ---
cox_model = joblib.load(".streamlit/cox_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- MongoDB connection ---
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# --- Page Setup and Styling ---
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")
st.markdown("""
<style>
h1 { color: #ad1457; text-align: center; font-weight: bold; }
.section-title {
    font-size: 22px;
    font-weight: bold;
    color: #ad1457;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.result-heading {
    font-size: 22px;
    color: #ad1457;
    font-weight: bold;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
.white-box {
    background-color: white;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    box-shadow: 0 0 5px rgba(0,0,0,0.1);
}
.stButton button {
    background-color: #ad1457 !important;
    color: white !important;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Breast Cancer Survival Prediction</h1>", unsafe_allow_html=True)

# --- Inputs ---
patient_id = st.text_input("Patient ID (Required)", key="patient_id")
if patient_id:
    prev = list(collection.find({"patient_id": patient_id}))
    if prev:
        with st.expander("Previous Predictions"):
            for r in prev:
                st.write(f"{r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ➔ 5yr: {r['survival_5yr']:.2f}, 10yr: {r['survival_10yr']:.2f}")

st.markdown("<div class='section-title'>Clinical Information</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age", key="age")
    lymph_nodes = st.text_input("Lymph Nodes Examined", key="nodes")
    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="meno")
    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="stage")
with col2:
    her2 = st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"], key="her2")
    er = st.selectbox("ER Status", ["", "Positive", "Negative"], key="er")
    pr = st.selectbox("PR Status", ["", "Positive", "Negative"], key="pr")

st.markdown("<div class='section-title'>Treatment Information</div>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    chemo = st.selectbox("Chemotherapy", ["", "Yes", "No"], key="chemo")
    surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"], key="surgery")
with col4:
    radio = st.selectbox("Radiotherapy", ["", "Yes", "No"], key="radio")
    hormone = st.selectbox("Hormone Therapy", ["", "Yes", "No"], key="hormone")

# --- Buttons ---
b1, b2 = st.columns(2)
with b1:
    if st.button("RESET"):
        st.query_params["reset"] = "1"
with b2:
    predict = st.button("PREDICT")

# --- Prediction Logic ---
if predict:
    required = [age, lymph_nodes, menopausal_status, er, pr, her2, chemo, radio, hormone, surgery, tumor_stage]
    if "" in required:
        st.error("Please fill out all fields before predicting.")
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

        collection.insert_one({
            "patient_id": patient_id,
            "timestamp": pd.Timestamp.now(),
            "survival_5yr": float(surv_5yr),
            "survival_10yr": float(surv_10yr)
        })

        # --- Results Container ---
        with st.container():
            st.markdown("<div class='white-box'>", unsafe_allow_html=True)
            st.markdown("<div class='result-heading'>Survival Predictions</div>", unsafe_allow_html=True)
            st.write(f"**5-Year Survival Probability:** {surv_5yr:.2f} ({surv_5yr * 100:.0f}%)")
            st.write(f"**10-Year Survival Probability:** {surv_10yr:.2f} ({surv_10yr * 100:.0f}%)")
            st.success("Patient record successfully saved!")
            st.markdown("</div>", unsafe_allow_html=True)

        # --- Results Overview ---
        st.markdown("<div class='section-title'>Results Overview</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            fig, ax = plt.subplots()
            ax.bar(["5-Year", "10-Year"], [surv_5yr, surv_10yr], color="#FF69B4")
            for i, v in enumerate([surv_5yr, surv_10yr]):
                ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        with c2:
            if surv_5yr < 0.5:
                st.error("Low Survival Chance")
                st.info("Patient shows low probability. Consider aggressive treatment planning.")
            elif surv_5yr < 0.75:
                st.warning("Moderate Survival Chance")
                st.info("Patient is at moderate risk. Monitor closely and adjust treatment accordingly.")
            else:
                st.success("High Survival Chance")
                st.info("Patient has a favorable survival outlook. Continue regular monitoring.")

        with c3:
            fig2, ax2 = plt.subplots()
            ax2.plot(times, surv_func.values.flatten(), color="#c2185b")
            ax2.set_title("Survival Curve")
            ax2.set_xlabel("Time (Months)")
            ax2.set_ylabel("Survival Probability")
            st.pyplot(fig2)

        # --- PDF Report ---
        pdf = BytesIO()
        c = canvas.Canvas(pdf, pagesize=letter)
        c.drawString(100, 750, f"Patient ID: {patient_id}")
        c.drawString(100, 730, f"5-Year Survival: {surv_5yr:.2f}")
        c.drawString(100, 710, f"10-Year Survival: {surv_10yr:.2f}")
        c.save()
        pdf.seek(0)
        st.download_button("Download Report", data=pdf, file_name=f"Survival_Report_{patient_id}.pdf", mime="application/pdf")

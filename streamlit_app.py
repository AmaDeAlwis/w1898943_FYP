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
from lifelines.plotting import plot_survival_function

# Load Cox model and scaler
cox_model = joblib.load(".streamlit/cox_model.pkl")
scaler = joblib.load("scaler.pkl")

# MongoDB connection
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# Custom CSS styling
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")
st.markdown("""
<style>
h1 {
    text-align: center;
    color: #880e4f;
    font-weight: bold;
}
.section-title {
    font-size: 20px;
    font-weight: bold;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
    color: #ad1457;
}
.stButton button {
    background-color: #ad1457 !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.5rem !important;
    margin-top: 1rem !important;
    text-transform: uppercase !important;
}
.white-container {
    background-color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Breast Cancer Survival Prediction</h1>", unsafe_allow_html=True)

# Initialize session state variables for form inputs
if "reset" not in st.session_state:
    st.session_state.reset = False

# Reset fields function
if st.session_state.reset:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.reset = False
    st.experimental_rerun()

# Input fields
st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    age = st.text_input("Age", key="age")
    lymph_nodes = st.text_input("Lymph Nodes Examined", key="lymph_nodes")
    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="menopausal")
    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4], key="tumor")
with col2:
    her2 = st.selectbox("HER2 Status", ["", "Gain", "Loss", "Neutral", "Undef"], key="her2")
    er = st.selectbox("ER Status", ["", "Positive", "Negative"], key="er")
    pr = st.selectbox("PR Status", ["", "Positive", "Negative"], key="pr")

# Live validation messages for age and lymph nodes
with col1:
    if age and (not age.isdigit() or int(age) < 20):
        st.warning("Age must be a number and ≥ 20")
    if lymph_nodes and (not lymph_nodes.isdigit() or int(lymph_nodes) < 0):
        st.warning("Lymph Nodes Examined must be a non-negative number")

st.markdown("<p class='section-title'>Treatment Information</p>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    chemo = st.selectbox("Chemotherapy", ["", "Yes", "No"], key="chemo")
    surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"], key="surgery")
with col4:
    radio = st.selectbox("Radiotherapy", ["", "Yes", "No"], key="radio")
    hormone = st.selectbox("Hormone Therapy", ["", "Yes", "No"], key="hormone")

# Patient ID input
patient_id = st.text_input("Patient ID (Required)", key="pid")
if patient_id:
    prev = list(collection.find({"patient_id": patient_id}))
    if prev:
        with st.expander("Previous Predictions"):
            for r in prev:
                st.write(f"{r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ➜ 5yr: {r['survival_5yr']:.2f}, 10yr: {r['survival_10yr']:.2f}")

# Buttons on the bottom
colb1, colb2 = st.columns(2)
with colb1:
    if st.button("RESET"):
        st.session_state.reset = True
        st.experimental_rerun()

with colb2:
    if st.button("PREDICT"):
        # Validation check
        required_fields = [age, lymph_nodes, menopausal_status, tumor_stage, her2, er, pr, chemo, radio, hormone, surgery, patient_id]
        if "" in required_fields or not age.isdigit() or not lymph_nodes.isdigit() or int(age) < 20 or int(lymph_nodes) < 0:
            st.error("Please fill all fields with valid inputs.")
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
                int(age), chemo_val, er_val, hormone_val, menopausal, int(lymph_nodes), pr_val, radio_val, int(tumor_stage),
                surgery_conserve, surgery_mastectomy, *her2_vals
            ]).reshape(1, -1)

            features_scaled = scaler.transform(features)
            df_input = pd.DataFrame(features_scaled, columns=cox_model.params_.index)
            surv_func = cox_model.predict_survival_function(df_input)
            times = surv_func.index.values
            surv_5yr = np.interp(60, times, surv_func.values.flatten())
            surv_10yr = np.interp(120, times, surv_func.values.flatten())

            st.markdown("""
                <div class="white-container">
                    <h4 style='color:#ad1457;'>Survival Predictions</h4>
                    <p><b>5-Year Survival Probability:</b> {0:.2f} ({1:.0f}%)</p>
                    <p><b>10-Year Survival Probability:</b> {2:.2f} ({3:.0f}%)</p>
                </div>
            """.format(surv_5yr, surv_5yr * 100, surv_10yr, surv_10yr * 100), unsafe_allow_html=True)

            # Save record
            record = {
                "patient_id": patient_id,
                "timestamp": pd.Timestamp.now(),
                "survival_5yr": float(surv_5yr),
                "survival_10yr": float(surv_10yr)
            }
            collection.insert_one(record)
            st.success("Patient record successfully saved!")

            # Plot results
            colres1, colres2, colres3 = st.columns(3)

            with colres1:
                fig, ax = plt.subplots()
                ax.bar(["5-Year", "10-Year"], [surv_5yr, surv_10yr], color="#FF69B4")
                ax.set_ylim(0, 1)
                for i, v in enumerate([surv_5yr, surv_10yr]):
                    ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
                st.pyplot(fig)

            with colres2:
                if surv_5yr < 0.5:
                    st.error("Low Survival Chance")
                    st.info("Patient shows low probability. Consider aggressive treatment planning.")
                else:
                    st.success("High Survival Chance")
                    st.info("Patient has a favorable survival outlook. Continue regular monitoring.")

            with colres3:
                fig2, ax2 = plt.subplots()
                ax2.plot(surv_func, color="#c2185b")
                ax2.set_title("Survival Curve")
                ax2.set_xlabel("Time (Months)")
                ax2.set_ylabel("Survival Probability")
                st.pyplot(fig2)

            # PDF Download
            pdf = BytesIO()
            c = canvas.Canvas(pdf, pagesize=letter)
            c.drawString(100, 750, f"Patient ID: {patient_id}")
            c.drawString(100, 730, f"5-Year Survival: {surv_5yr:.2f}")
            c.drawString(100, 710, f"10-Year Survival: {surv_10yr:.2f}")
            c.save()
            pdf.seek(0)
            st.download_button("Download Report", data=pdf, file_name=f"Survival_Report_{patient_id}.pdf", mime="application/pdf")

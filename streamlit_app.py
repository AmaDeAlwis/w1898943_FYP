import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pymongo import MongoClient
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import matplotlib.pyplot as plt

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
    h1 {
        text-align: center;
        color: #ad1457;
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
        background-color: #ad1457;
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    .prediction-box {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>Breast Cancer Survival Prediction</h1>", unsafe_allow_html=True)

# --- Reset function ---
def reset_fields():
    for k in st.session_state.keys():
        del st.session_state[k]

# --- Buttons ---
col_a, col_b = st.columns(2)
with col_a:
    if st.button("RESET"):
        reset_fields()
        st.experimental_rerun()

# --- Input Fields ---
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
    age = st.text_input("Age", key="age")
    if age and (not age.isdigit() or int(age) < 20):
        st.warning("Age must be a number and ≥ 20")

    lymph_nodes = st.text_input("Lymph Nodes Examined", key="ln")
    if lymph_nodes and (not lymph_nodes.lstrip('-').isdigit() or int(lymph_nodes) < 0):
        st.warning("Lymph Nodes Examined must be a non-negative number")

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

# --- Predict button ---
with col_b:
    predict = st.button("PREDICT")

if predict:
    if "" in [age, lymph_nodes, menopausal_status, er, pr, her2, chemo, radio, hormone, surgery, tumor_stage]:
        st.error("Please fill all the required fields before predicting.")
    else:
        # Preprocessing
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
            int(age), chemo, er, hormone, menopausal, int(lymph_nodes), pr, radio, int(tumor_stage),
            surgery_conserve, surgery_mastectomy, *her2_vals
        ]).reshape(1, -1)

        features_scaled = scaler.transform(features)
        df_input = pd.DataFrame(features_scaled, columns=cox_model.params_.index)
        surv_func = cox_model.predict_survival_function(df_input)
        times = surv_func.index.values
        surv_5yr = np.interp(60, times, surv_func.values.flatten())
        surv_10yr = np.interp(120, times, surv_func.values.flatten())

        record = {
            "patient_id": patient_id,
            "timestamp": pd.Timestamp.now(),
            "survival_5yr": float(surv_5yr),
            "survival_10yr": float(surv_10yr)
        }
        collection.insert_one(record)
        st.success("Patient record successfully saved!")

        # Display Results
        st.markdown("""
            <div class='prediction-box'>
            <h4 style='color:#ad1457'>Survival Predictions</h4>
            <p><strong>5-Year Survival Probability:</strong> {:.2f} ({:.0f}%)</p>
            <p><strong>10-Year Survival Probability:</strong> {:.2f} ({:.0f}%)</p>
            </div>
        """.format(surv_5yr, surv_5yr*100, surv_10yr, surv_10yr*100), unsafe_allow_html=True)

        st.markdown("<h4 class='section-title'>Results Overview</h4>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            fig, ax = plt.subplots()
            ax.bar(["5-Year", "10-Year"], [surv_5yr, surv_10yr], color="#FF69B4")
            ax.set_ylim(0, 1)
            for i, v in enumerate([surv_5yr, surv_10yr]):
                ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
            st.pyplot(fig)
        with c2:
            if surv_5yr < 0.5:
                st.error("Low Survival Chance")
                st.info("Patient shows low probability. Consider aggressive treatment planning.")
            else:
                st.success("High Survival Chance")
                st.info("Patient has a favorable survival outlook. Continue regular monitoring.")
        with c3:
            fig, ax = plt.subplots()
            ax.plot(times, surv_func.values.flatten(), color="#ad1457")
            ax.set_xlabel("Time (Months)")
            ax.set_ylabel("Survival Probability")
            ax.set_title("Survival Curve")
            st.pyplot(fig)

        # PDF Download
        pdf = BytesIO()
        c = canvas.Canvas(pdf, pagesize=letter)
        c.drawString(100, 750, f"Patient ID: {patient_id}")
        c.drawString(100, 730, f"5-Year Survival: {surv_5yr:.2f} ({surv_5yr*100:.0f}%)")
        c.drawString(100, 710, f"10-Year Survival: {surv_10yr:.2f} ({surv_10yr*100:.0f}%)")
        c.save()
        pdf.seek(0)

        st.download_button("Download Report", data=pdf, file_name=f"Survival_Report_{patient_id}.pdf", mime="application/pdf")

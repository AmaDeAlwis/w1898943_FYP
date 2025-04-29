# --- FINAL Streamlit Code Fixing Everything You Asked ---

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

# --- Load Cox model and scaler ---
cox_model = joblib.load(".streamlit/cox_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- MongoDB connection ---
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# --- Page configuration ---
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
<style>
h1 { color: #ad1457; text-align: center; font-weight: bold; }
.section-title { font-size: 22px; font-weight: bold; margin-top: 2rem; margin-bottom: 0.5rem; color: #ad1457; }
.result-heading { font-size: 24px; color: #ad1457; font-weight: bold; margin-bottom: 1rem; }
.stButton button { background-color: #ad1457 !important; color: white !important; border-radius: 10px; font-weight: bold; }
.white-container { background-color: white; padding: 20px; border-radius: 10px; }
.metric-title { font-weight: bold; font-size: 18px; color: #c2185b; }
</style>
""", unsafe_allow_html=True)

# --- Main Title ---
st.markdown("<h1> Breast Cancer Survival Prediction </h1>", unsafe_allow_html=True)

# --- Patient ID ---
patient_id = st.text_input("Patient ID (Required)")
if patient_id:
    prev = list(collection.find({"patient_id": patient_id}))
    if prev:
        with st.expander("Previous Predictions"):
            for r in prev:
                st.write(f"{r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ➔ 5yr: {r['survival_5yr']:.2f}, 10yr: {r['survival_10yr']:.2f}")

# --- Input Fields ---
st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    age = st.text_input("Age", key="age")
    if age:
        try:
            age_val = float(age)
            if age_val < 20:
                st.warning("Age must be a number and ≥ 20", icon="⚠️")
        except:
            st.warning("Age must be a valid number", icon="⚠️")

    lymph_nodes = st.text_input("Lymph Nodes Examined", key="nodes")
    if lymph_nodes:
        try:
            nodes_val = float(lymph_nodes)
            if nodes_val < 0:
                st.warning("Lymph Nodes Examined must be non-negative", icon="⚠️")
        except:
            st.warning("Lymph Nodes Examined must be a valid number", icon="⚠️")

    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"], key="menopause")
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

# --- Buttons ---
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    reset = st.button("RESET")
with col_btn2:
    predict = st.button("PREDICT")

# --- RESET BUTTON ---
if reset:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# --- PREDICT BUTTON ---
if predict:
    required_fields = [age, lymph_nodes, menopausal_status, tumor_stage, her2, er, pr, chemo, radio, hormone, surgery]
    if "" in required_fields or None in required_fields:
        st.error("❌ Please fill out all fields before predicting.")
    else:
        try:
            features = np.array([
                float(age),
                1 if chemo == "Yes" else 0,
                1 if er == "Positive" else 0,
                1 if hormone == "Yes" else 0,
                1 if menopausal_status == "Post-menopausal" else 0,
                float(lymph_nodes),
                1 if pr == "Positive" else 0,
                1 if radio == "Yes" else 0,
                int(tumor_stage),
                1 if surgery == "Breast-conserving" else 0,
                1 if surgery == "Mastectomy" else 0,
                *(1 if her2 == opt else 0 for opt in ["Gain", "Loss", "Neutral", "Undef"])
            ]).reshape(1, -1)

            features_scaled = scaler.transform(features)
            df_input = pd.DataFrame(features_scaled, columns=cox_model.params_.index)
            surv_func = cox_model.predict_survival_function(df_input)
            times = surv_func.index.values
            surv_5yr = np.interp(60, times, surv_func.values.flatten())
            surv_10yr = np.interp(120, times, surv_func.values.flatten())

            # --- White Container for Predictions ---
            with st.container():
                st.markdown("<div class='white-container'>", unsafe_allow_html=True)
                st.markdown("<h2 class='result-heading'>Survival Predictions</h2>", unsafe_allow_html=True)
                st.markdown(f"5-Year Survival Probability: **{surv_5yr:.2f} ({surv_5yr*100:.0f}%)**")
                st.markdown(f"10-Year Survival Probability: **{surv_10yr:.2f} ({surv_10yr*100:.0f}%)**")
                st.markdown("</div>", unsafe_allow_html=True)

            # --- Save to MongoDB ---
            record = {
                "patient_id": patient_id,
                "timestamp": pd.Timestamp.now(),
                "survival_5yr": float(surv_5yr),
                "survival_10yr": float(surv_10yr)
            }
            collection.insert_one(record)
            st.success("📅 Patient record successfully saved!")

            # --- Results Overview ---
            st.markdown("<h2 class='result-heading'>Results Overview</h2>", unsafe_allow_html=True)
            
            left, center, right = st.columns([1, 1, 1])
            with left:
                fig, ax = plt.subplots()
                ax.bar(["5-Year", "10-Year"], [surv_5yr, surv_10yr], color="#FF69B4")
                for i, val in enumerate([surv_5yr, surv_10yr]):
                    ax.text(i, val + 0.02, f"{val:.2f}", ha='center')
                ax.set_ylim(0, 1)
                st.pyplot(fig)

            with center:
                if surv_5yr >= 0.8:
                    st.success("High Survival Chance")
                    st.info("Patient has a favorable survival outlook. Continue regular monitoring.")
                else:
                    st.error("Low Survival Chance")
                    st.warning("Patient shows low probability. Consider aggressive treatment planning.")

            with right:
                fig2, ax2 = plt.subplots()
                ax2.plot(times, surv_func.values.flatten(), color="#C71585")
                ax2.set_xlabel("Time (Months)")
                ax2.set_ylabel("Survival Probability")
                ax2.set_title("Survival Curve")
                st.pyplot(fig2)

            # --- PDF Report Generation ---
            pdf = BytesIO()
            c = canvas.Canvas(pdf, pagesize=letter)
            c.drawString(100, 750, f"Patient ID: {patient_id}")
            c.drawString(100, 730, f"5-Year Survival: {surv_5yr:.2f} ({surv_5yr*100:.0f}%)")
            c.drawString(100, 710, f"10-Year Survival: {surv_10yr:.2f} ({surv_10yr*100:.0f}%)")
            c.save()
            pdf.seek(0)
            st.download_button("Download Report", data=pdf, file_name=f"Survival_Report_{patient_id}.pdf", mime="application/pdf")

        except Exception as e:
            st.error(f"An error occurred: {e}")

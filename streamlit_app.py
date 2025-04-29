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

# Load model and scaler
cox_model = joblib.load(".streamlit/cox_model.pkl")
scaler = joblib.load("scaler.pkl")

# MongoDB connection
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# Set layout and custom style
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")
st.markdown("""
<style>
    h1 { color: #ad1457; text-align: center; font-weight: bold; font-size: 36px; }
    .section-title { font-size: 22px; font-weight: bold; color: #ad1457; margin-top: 2rem; }
    .result-container { background-color: white; border-radius: 12px; padding: 1rem 2rem; margin-bottom: 1rem; }
    .result-title { color: #ad1457; font-size: 24px; font-weight: bold; }
    .result-subtext { color: #000; font-size: 18px; }
    .stButton button {
        background-color: #ad1457 !important;
        color: white !important;
        border-radius: 10px;
        font-weight: bold;
        padding: 0.6rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Breast Cancer Survival Prediction</h1>", unsafe_allow_html=True)

# Session state defaults
for key in ["age", "lymph_nodes"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# --- INPUT FORM ---
with st.form("prediction_form"):
    st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        age = st.text_input("Age", value=st.session_state.age, key="age")
        if age and (not age.isdigit() or int(age) < 20):
            st.warning("Age must be a number and ≥ 20")

        lymph_nodes = st.text_input("Lymph Nodes Examined", value=st.session_state.lymph_nodes, key="lymph_nodes")
        if lymph_nodes and (not lymph_nodes.isdigit() or int(lymph_nodes) < 0):
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

    # Buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        reset = st.form_submit_button("RESET")
    with col_btn2:
        predict = st.form_submit_button("PREDICT")

# --- RESET FUNCTION ---
def reset_fields():
    for field in ["age", "lymph_nodes"]:
        st.session_state[field] = ""
    st.experimental_rerun()

# Handle reset
if reset:
    reset_fields()

# --- PREDICT ---
if predict:
    required_fields = [age, lymph_nodes, menopausal_status, tumor_stage, er, pr, her2, chemo, radio, hormone, surgery]
    if "" in required_fields:
        st.error("❗ Please fill out all required fields before predicting.")
    elif not age.isdigit() or int(age) < 20:
        st.error("❗ Age must be a number and ≥ 20.")
    elif not lymph_nodes.isdigit() or int(lymph_nodes) < 0:
        st.error("❗ Lymph Nodes Examined must be a non-negative number.")
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
            int(age), chemo, er, hormone, menopausal, int(lymph_nodes), pr, radio, int(tumor_stage),
            surgery_conserve, surgery_mastectomy, *her2_vals
        ]).reshape(1, -1)

        features_scaled = scaler.transform(features)
        df_input = pd.DataFrame(features_scaled, columns=cox_model.params_.index)

        surv_func = cox_model.predict_survival_function(df_input)
        times = surv_func.index.values
        surv_5yr = float(np.interp(60, times, surv_func.values.flatten()))
        surv_10yr = float(np.interp(120, times, surv_func.values.flatten()))

        # Save to DB
        collection.insert_one({
            "timestamp": pd.Timestamp.now(),
            "survival_5yr": surv_5yr,
            "survival_10yr": surv_10yr
        })

        st.success("Patient record successfully saved!")

        # --- DISPLAY RESULTS ---
        st.markdown("""
        <div class='result-container'>
            <p class='result-title'>Survival Predictions</p>
            <p class='result-subtext'>5-Year Survival Probability: <b>{:.2f}</b> ({}%)</p>
            <p class='result-subtext'>10-Year Survival Probability: <b>{:.2f}</b> ({}%)</p>
        </div>
        """.format(surv_5yr, int(surv_5yr * 100), surv_10yr, int(surv_10yr * 100)), unsafe_allow_html=True)

        # --- RESULTS OVERVIEW ---
        st.markdown("<p class='section-title'>Results Overview</p>", unsafe_allow_html=True)
        colres1, colres2, colres3 = st.columns(3)

        with colres1:
            fig, ax = plt.subplots()
            ax.bar(["5-Year", "10-Year"], [surv_5yr, surv_10yr], color="#FF69B4")
            ax.set_ylim(0, 1)
            for i, v in enumerate([surv_5yr, surv_10yr]):
                ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
            st.pyplot(fig)

        with colres2:
            if surv_5yr < 0.5:
                st.error("Low Survival Chance")
                st.info("Patient shows low probability. Consider aggressive treatment planning.")
            else:
                st.success("High Survival Chance")
                st.info("Patient has a favorable survival outlook. Continue regular monitoring.")

        with colres3:
            fig, ax = plt.subplots()
            plot_survival_function(surv_func, ax=ax, color="#c2185b")
            ax.set_title("Survival Curve")
            ax.set_xlabel("Time (Months)")
            ax.set_ylabel("Survival Probability")
            st.pyplot(fig)

        # --- DOWNLOAD ---
        pdf = BytesIO()
        c = canvas.Canvas(pdf, pagesize=letter)
        c.drawString(100, 750, f"5-Year Survival: {surv_5yr:.2f} ({int(surv_5yr * 100)}%)")
        c.drawString(100, 730, f"10-Year Survival: {surv_10yr:.2f} ({int(surv_10yr * 100)}%)")
        c.save()
        pdf.seek(0)
        st.download_button("Download Report", data=pdf, file_name="Survival_Report.pdf", mime="application/pdf")

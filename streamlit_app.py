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

# --- Keys for resetting ---
reset_keys = ["patient_id", "age", "nodes", "meno", "stage", "her2", "er", "pr", "chemo", "surgery", "radio", "hormone"]

# --- Default values for form inputs ---
if st.session_state.get("reset_triggered"):
    default_values = {k: "" for k in reset_keys}
    st.session_state.pop("reset_triggered")
else:
    default_values = {k: st.session_state.get(k, "") for k in reset_keys}

# --- Load model and scaler ---
cox_model = joblib.load(".streamlit/cox_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- MongoDB connection ---
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# --- Page setup ---
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
    margin-top: 1rem;
    margin-bottom: 2rem;
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

# --- Patient ID ---
patient_id = st.text_input("Patient ID (Required)", value=default_values["patient_id"], key="patient_id")

# --- Inputs ---
st.markdown("<div class='section-title'>Clinical Information</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age", value=default_values["age"], key="age")
    lymph_nodes = st.text_input("Lymph Nodes Examined", value=default_values["nodes"], key="nodes")
    meno_opts = ["", "Pre-menopausal", "Post-menopausal"]
    menopausal_status = st.selectbox("Menopausal Status", meno_opts,
                                     index=meno_opts.index(default_values["meno"]) if default_values["meno"] in meno_opts else 0,
                                     key="meno")
    stage_opts = ["", 1, 2, 3, 4]
    tumor_stage = st.selectbox("Tumor Stage", stage_opts,
                               index=stage_opts.index(int(default_values["stage"])) if str(default_values["stage"]).isdigit() and int(default_values["stage"]) in stage_opts else 0,
                               key="stage")

with col2:
    her2_opts = ["", "Neutral", "Loss", "Gain", "Undef"]
    her2 = st.selectbox("HER2 Status", her2_opts,
                        index=her2_opts.index(default_values["her2"]) if default_values["her2"] in her2_opts else 0,
                        key="her2")
    er_opts = ["", "Positive", "Negative"]
    er = st.selectbox("ER Status", er_opts,
                      index=er_opts.index(default_values["er"]) if default_values["er"] in er_opts else 0,
                      key="er")
    pr_opts = ["", "Positive", "Negative"]
    pr = st.selectbox("PR Status", pr_opts,
                      index=pr_opts.index(default_values["pr"]) if default_values["pr"] in pr_opts else 0,
                      key="pr")

st.markdown("<div class='section-title'>Treatment Information</div>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    chemo_opts = ["", "Yes", "No"]
    chemo = st.selectbox("Chemotherapy", chemo_opts,
                         index=chemo_opts.index(default_values["chemo"]) if default_values["chemo"] in chemo_opts else 0,
                         key="chemo")
    surgery_opts = ["", "Breast-conserving", "Mastectomy"]
    surgery = st.selectbox("Surgery Type", surgery_opts,
                           index=surgery_opts.index(default_values["surgery"]) if default_values["surgery"] in surgery_opts else 0,
                           key="surgery")
with col4:
    radio_opts = ["", "Yes", "No"]
    radio = st.selectbox("Radiotherapy", radio_opts,
                         index=radio_opts.index(default_values["radio"]) if default_values["radio"] in radio_opts else 0,
                         key="radio")
    hormone_opts = ["", "Yes", "No"]
    hormone = st.selectbox("Hormone Therapy", hormone_opts,
                           index=hormone_opts.index(default_values["hormone"]) if default_values["hormone"] in hormone_opts else 0,
                           key="hormone")

# --- Buttons ---
col_b1, col_b2 = st.columns(2)
with col_b1:
    if st.button("RESET"):
        st.session_state["reset_triggered"] = True
        st.rerun()
with col_b2:
    if st.button("PREDICT"):
        if "" in [age, lymph_nodes, menopausal_status, er, pr, her2, chemo, radio, hormone, surgery, tumor_stage]:
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

            st.success("✅ Prediction complete and saved!")
            st.markdown("<div class='white-box'>", unsafe_allow_html=True)
            st.markdown("<div class='result-heading'>Survival Predictions</div>", unsafe_allow_html=True)
            st.write(f"**5-Year Survival Probability:** {surv_5yr:.2f} ({surv_5yr * 100:.0f}%)")
            st.write(f"**10-Year Survival Probability:** {surv_10yr:.2f} ({surv_10yr * 100:.0f}%)")
            st.markdown("</div>", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pymongo import MongoClient
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Breast Cancer Survival Prediction</h1>", unsafe_allow_html=True)

# --- Inputs ---
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
    st.button("PREDICT")

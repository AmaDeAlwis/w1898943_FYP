import streamlit as st
import pandas as pd
import numpy as np

# ---- Page Configuration ----
st.set_page_config(page_title="Breast Cancer UI", layout="centered")

# ---- Custom CSS ----
st.markdown("""
    <style>
        .main-container {
            background-color: white;
            padding: 2rem;
            border-radius: 30px;
            margin: 1rem auto;
            width: 95vw;
            max-width: 1200px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
        }
        .pink-box {
            background-color: #f9b3c2;
            padding: 3rem;
            border-radius: 25px;
            height: 100%;
        }
        h1 {
            text-align: center;
            color: #C2185B;
        }
        .stApp > header, .stApp > footer {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Main Layout ----
st.markdown('<div class="main-container"><div class="pink-box">', unsafe_allow_html=True)

# ---- Title ----
st.markdown("<h1>🎀 Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)
st.write("Fill in the details below to generate predictions and insights.")

# ---- Form Layout ----
with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=20, max_value=90, value=50)
        menopausal_status = st.selectbox("Menopausal Status", ["Pre-menopausal", "Post-menopausal"])
        tumor_stage = st.selectbox("Tumor Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
        lymph_nodes_examined = st.number_input("Lymph Nodes Examined", min_value=0, max_value=50, value=3)
    with col2:
        er_status = st.selectbox("ER Status", ["Positive", "Negative"])
        pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
        her2_status = st.selectbox("HER2 Status", ["Positive", "Negative"])

    col3, col4 = st.columns(2)
    with col3:
        chemotherapy = st.selectbox("Chemotherapy", ["Yes", "No"])
        surgery = st.selectbox("Surgery Type", ["Breast-conserving", "Mastectomy"])
    with col4:
        radiotherapy = st.selectbox("Radiotherapy", ["Yes", "No"])
        hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])

    st.form_submit_button("🔍 Predict")

# ---- Close Layout ----
st.markdown('</div></div>', unsafe_allow_html=True)

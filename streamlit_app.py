import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import matplotlib.pyplot as plt

st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# ---- Custom CSS ----
def apply_custom_styles():
    st.markdown("""
    <style>
        .main {
            padding: 40px 80px;
            background-color: #f9b3c2;
            border: 8px solid white;
            border-radius: 20px;
        }
        h1 {
            text-align: center;
            color: #C2185B;
        }
        .form-block {
            background-color: #ffc3d9;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# ---- Title ----
st.markdown("<h1>🎀 Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)
st.markdown("#### Fill in the details below to generate predictions and insights.")

# ---- Form ----
with st.form("patient_form"):
    # Clinical Data Block
    st.markdown("### 🧬 Clinical Data")
    st.markdown('<div class="form-block">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=20, max_value=90, value=50)
        tumor_stage = st.selectbox("Tumor Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
        menopausal_status = st.selectbox("Menopausal Status", ["Pre-menopausal", "Post-menopausal"])
        lymph_nodes_examined = st.number_input("Lymph Nodes Examined", min_value=0, max_value=50, value=3)

    with col2:
        pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
        her2_status = st.selectbox("HER2 Status", ["Positive", "Negative"])
        er_status = st.selectbox("ER Status", ["Positive", "Negative"])

    st.markdown('</div>', unsafe_allow_html=True)

    # Treatment Data Block
    st.markdown("### 💊 Treatment Details")
    st.markdown('<div class="form-block">', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        chemotherapy = st.selectbox("Chemotherapy", ["Yes", "No"])
        surgery = st.selectbox("Surgery Type", ["Breast-conserving", "Mastectomy"])

    with col4:
        radiotherapy = st.selectbox("Radiotherapy", ["Yes", "No"])
        hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])

    st.markdown('</div>', unsafe_allow_html=True)

    # Buttons (bottom left)
    col_btn_left, col_btn_space, _ = st.columns([1, 6, 1])
    with col_btn_left:
        reset = st.form_submit_button("🔄 RESET")
        submit = st.form_submit_button("🔍 PREDICT")

# ---- Reset Logic ----
if reset:
    st.experimental_rerun()

# ---- Prediction Logic ----
if submit:
    # Collect input
    user_data = {
        "Age": age,
        "Tumor_Stage": tumor_stage,
        "Menopausal_Status": menopausal_status,
        "Lymph_Nodes_Examined": lymph_nodes_examined,
        "ER_Status": er_status,
        "PR_Status": pr_status,
        "HER2_Status": her2_status,
        "Surgery_Type": surgery,
        "Chemotherapy": chemotherapy,
        "Radiotherapy": radiotherapy,
        "Hormone_Therapy": hormone_therapy
    }

    # Simulate prediction
    probability = np.random.uniform(0.6, 0.95)
    st.success(f"🧬 Estimated 5-Year Survival Probability: **{probability:.2%}**")

    # Visualization
    st.subheader("📊 Visual Summary")
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.barh(["Survival Probability"], [probability], color="#d63384")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)

    # Save to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["breast_cancer_app"]
    collection = db["patient_inputs"]
    collection.insert_one({**user_data, "Survival_Probability": probability})
    st.success("✅ Data saved to local MongoDB!")

# ---- Footer ----
st.markdown("---")
st.caption("Created with ❤️ to support breast cancer awareness")

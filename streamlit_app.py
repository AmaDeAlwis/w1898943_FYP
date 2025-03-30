import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import matplotlib.pyplot as plt

st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")



# 🎨 Then apply custom CSS
def styled_input_form():
    st.markdown("""
    <style>
    .form-container {
        background-color: #FFE6EB;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    .form-container label {
        font-weight: bold;
        color: #C2185B;
    }
    </style>
    """, unsafe_allow_html=True)

styled_input_form()

# ----- 🔧 MongoDB Setup -----
def save_to_mongo(data):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["breast_cancer_app"]
    collection = db["patient_inputs"]
    collection.insert_one(data)

# ----- 🩺 Title & Layout -----
st.title(" Breast Cancer Survival Prediction Interface")

st.markdown("Fill in the details below to generate predictions and insights.")

# ----- 📋 Input Form -----
with st.form("patient_form"):
    st.markdown("## 🧬 Clinical Data")
    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=20, max_value=90, value=50)
        er_status = st.selectbox("ER Status", ["Positive", "Negative"])
        pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
        her2_status = st.selectbox("HER2 Status", ["Positive", "Negative"])
        menopausal_status = st.selectbox("Menopausal Status", ["Pre-menopausal", "Post-menopausal"])

    with col2:
        tumor_stage = st.selectbox("Tumor Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
        lymph_nodes_examined = st.number_input("Lymph Nodes Examined", min_value=0, max_value=50, value=3)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Treatment Section ---
    st.markdown("## 🩺 Treatment Details")
    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        surgery = st.selectbox("Surgery Type", ["Breast-conserving", "Mastectomy"])
        chemotherapy = st.selectbox("Chemotherapy", ["Yes", "No"])

    with col4:
        radiotherapy = st.selectbox("Radiotherapy", ["Yes", "No"])
        hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])

    st.markdown('</div>', unsafe_allow_html=True)

    # Buttons
    reset_btn, predict_btn = st.columns([1, 3])
    submitted = predict_btn.form_submit_button("PREDICT")
    reset = reset_btn.form_submit_button("RESET")


# ----- 🔁 Reset Logic -----
if reset:
    st.experimental_rerun()

# ----- 🔮 Prediction & Visualization -----
if submitted:
    user_data = {
        "Age": age,
        "ER_Status": er_status,
        "PR_Status": pr_status,
        "HER2_Status": her2_status,
        "Menopausal_Status": menopausal_status,
        "Tumor_Stage": tumor_stage,
        "Lymph_Nodes_Examined": lymph_nodes_examined,
        "Surgery_Type": surgery,
        "Chemotherapy": chemotherapy,
        "Radiotherapy": radiotherapy,
        "Hormone_Therapy": hormone_therapy
    }

    # ---- Simulated Prediction ----
    probability = np.random.uniform(0.6, 0.95)
    st.success(f"🧬 Estimated 5-Year Survival Probability: **{probability:.2%}**")

    # ---- Basic Visualization ----
    st.subheader("📊 Visual Summary")
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.barh(["Survival Probability"], [probability], color="#d63384")
    ax.set_xlim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    st.pyplot(fig)

    # ---- Save to MongoDB ----
    save_to_mongo({**user_data, "Survival_Probability": probability})
    st.success("✅ Data saved to local MongoDB!")

# Footer
st.markdown("---")
st.caption("Created with ❤️ for breast cancer survival awareness")

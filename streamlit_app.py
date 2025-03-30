import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# ----- 💅 CSS Styling -----
st.markdown("""
<style>
body {
    background-color: white;
}
.page-wrapper {
    display: flex;
    justify-content: center;
    padding: 3rem 0;
}
.bordered-container {
    background-color: #f9b3c2;
    padding: 3rem;
    border-radius: 30px;
    border: 8px solid white;
    width: 90%;
    box-shadow: 0 0 15px rgba(0,0,0,0.05);
}
.section {
    background-color: #ffe0eb;
    padding: 25px;
    border-radius: 15px;
    margin-top: 30px;
    margin-bottom: 30px;
}
.section h3 {
    margin-top: 0;
    color: #C2185B;
}
h1 {
    text-align: center;
    color: #C2185B;
}
</style>
""", unsafe_allow_html=True)

# ----- 🚫 Remove ALL old container boxes above this -----

# ----- ✅ Page Wrapper Starts -----
st.markdown('<div class="page-wrapper"><div class="bordered-container">', unsafe_allow_html=True)

# ----- 🎀 Title & Description -----
st.markdown("<h1>🎀 Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)
st.markdown("#### Fill in the details below to generate predictions and insights.")

# ----- 📝 Form Starts -----
with st.form("patient_form"):

    # ----- Clinical Data -----
    st.markdown('<div class="section"><h3>🧬 Clinical Data</h3>', unsafe_allow_html=True)

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

    # ----- Treatment Data -----
    st.markdown('<div class="section"><h3>💊 Treatment Details</h3>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        chemotherapy = st.selectbox("Chemotherapy", ["Yes", "No"])
        surgery = st.selectbox("Surgery Type", ["Breast-conserving", "Mastectomy"])
    with col4:
        radiotherapy = st.selectbox("Radiotherapy", ["Yes", "No"])
        hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])

    st.markdown('</div>', unsafe_allow_html=True)

    # ----- Buttons -----
    col_reset, col_submit = st.columns([1, 6])
    with col_reset:
        reset = st.form_submit_button("🔄 RESET")
    with col_submit:
        submit = st.form_submit_button("🔍 PREDICT")

# ----- 🔁 Reset Logic -----
if reset:
    st.experimental_rerun()

# ----- 🔮 Prediction Logic -----
if submit:
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

    # Simulate survival probability
    probability = np.random.uniform(0.6, 0.95)
    st.success(f"🧬 Estimated 5-Year Survival Probability: **{probability:.2%}**")

    # Visual summary
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

# ----- ❌ Close Page Wrapper -----
st.markdown('</div></div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Created with ❤️ to support breast cancer awareness")

import streamlit as st

# Page config
st.set_page_config(page_title="Breast Cancer UI", layout="centered")

# Custom CSS
st.markdown("""
<style>
body {
    background-color: white;
}
.main-box {
    background-color: #f9b3c2;
    padding: 3rem;
    border-radius: 25px;
    width: 95%;
    max-width: 1100px;
    margin: 2rem auto;
}
h1 {
    text-align: center;
    color: #000000;
}
p {
    text-align: center;
}
.stTextInput > div > input,
.stSelectbox > div > div,
.stNumberInput > div > input {
    background-color: #ffc3d9;
    border-radius: 10px;
}
.stButton>button {
    background-color: #d63384;
    color: white;
    border: none;
    border-radius: 12px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Main pink container
st.markdown('<div class="main-box">', unsafe_allow_html=True)

# Title
st.markdown("### 🎀 Breast Cancer Survival Prediction Interface")
st.markdown("Fill in the details below to generate predictions and insights.")

# Form
with st.form("form"):
    st.markdown("#### 🧬 Clinical Data")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=20, max_value=90, value=50)
        menopausal_status = st.selectbox("Menopausal Status", ["Pre-menopausal", "Post-menopausal"])
        tumor_stage = st.selectbox("Tumor Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
        lymph_nodes = st.number_input("Lymph Nodes Examined", min_value=0, max_value=50, value=3)
    with col2:
        er_status = st.selectbox("ER Status", ["Positive", "Negative"])
        pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
        her2_status = st.selectbox("HER2 Status", ["Positive", "Negative"])

    st.markdown("#### 💊 Treatment Data")
    col3, col4 = st.columns(2)
    with col3:
        chemotherapy = st.selectbox("Chemotherapy", ["Yes", "No"])
        surgery = st.selectbox("Surgery Type", ["Breast-conserving", "Mastectomy"])
    with col4:
        radiotherapy = st.selectbox("Radiotherapy", ["Yes", "No"])
        hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])

    col5, col6 = st.columns(2)
    with col5:
        reset = st.form_submit_button("🔄 Reset")
    with col6:
        submit = st.form_submit_button("🔍 Predict")

    if reset:
        st.experimental_rerun()
    if submit:
        st.success("✅ Prediction done! (Simulated)")

# Close main box
st.markdown("</div>", unsafe_allow_html=True)

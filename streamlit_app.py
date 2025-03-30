import streamlit as st

st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# ----- Custom CSS -----
st.markdown("""
<style>
body {
    background-color: white;
}

.main-container {
    background-color: #f9b3c2;
    border: 8px solid white;
    border-radius: 25px;
    padding: 3rem 3rem 5rem 3rem;
    margin: 2rem auto;
    width: 90%;
    max-width: 1200px;
    box-shadow: 0 0 15px rgba(0,0,0,0.1);
}

h1 {
    color: black;
    text-align: center;
    font-size: 2.5rem;
}

h4 {
    text-align: center;
    color: #333;
}

.section {
    background-color: #ffc3d9;
    padding: 20px;
    border-radius: 15px;
    margin-top: 30px;
    margin-bottom: 20px;
}

.stButton>button {
    background-color: #d63384;
    color: white;
    font-weight: bold;
    padding: 0.75rem 2rem;
    border-radius: 10px;
    border: none;
    width: 100%;
}

input, select {
    background-color: #ffe0eb !important;
    border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)

# ----- Start UI -----
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ----- Title -----
st.markdown("## 🎀 Breast Cancer Survival Prediction Interface")
st.markdown("<h4>Fill in the details below to generate predictions and insights.</h4>", unsafe_allow_html=True)

# ---- FORM ----
with st.form("patient_form"):
    # Clinical Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 🧬 Clinical Data")
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
    st.markdown('</div>', unsafe_allow_html=True)

    # Treatment Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 💊 Treatment Data")
    col3, col4 = st.columns(2)
    with col3:
        chemotherapy = st.selectbox("Chemotherapy", ["Yes", "No"])
        surgery = st.selectbox("Surgery Type", ["Breast-conserving", "Mastectomy"])
    with col4:
        radiotherapy = st.selectbox("Radiotherapy", ["Yes", "No"])
        hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        reset = st.form_submit_button("🔄 RESET")
    with col_btn2:
        submit = st.form_submit_button("🔍 PREDICT")

    if reset:
        st.experimental_rerun()
    if submit:
        st.success("✅ Prediction complete (simulated)!")

# ----- Close Container -----
st.markdown('</div>', unsafe_allow_html=True)

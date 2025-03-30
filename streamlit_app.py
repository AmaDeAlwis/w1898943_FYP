import streamlit as st

# Set wide layout
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
h1 {
    text-align: center;
    color: #FFFFFF;
}

.section-title {
    font-size: 20px;
    font-weight: bold;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
    color: #ad1457;
}

.stButton>button {
    background-color: #d63384;
    color: white;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    margin-top: 1rem;
    cursor: pointer;
}

input, select, textarea {
    border-radius: 10px !important;
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

# --- UI Layout ---
st.markdown('<div class="container">', unsafe_allow_html=True)

st.markdown("<h1> Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Fill in the details below to generate predictions and insights.</p>", unsafe_allow_html=True)

# Form Start
with st.form("input_form"):
    st.markdown("<div class='section-title'> Clinical Data</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=20, max_value=96)
        menopausal_status = st.selectbox("Menopausal Status", ["Pre-menopausal", "Post-menopausal"])
        tumor_stage = st.selectbox("Tumor Stage", [1, 2, 3, 4])
        lymph_nodes_examined = st.number_input("Lymph Nodes Examined", min_value=0, max_value=50)

    with col2:
        er_status = st.selectbox("ER Status", ["Positive", "Negative"])
        pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
        her2_status = st.selectbox("HER2 Status", ["Neutral", "Loss", "Gain", "Undef"])

    st.markdown("<div class='section-title'> Treatment Data</div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        chemotherapy = st.selectbox("Chemotherapy", ["Yes", "No"])
        surgery = st.selectbox("Surgery Type", ["Breast-conserving", "Mastectomy"])
    with col4:
        radiotherapy = st.selectbox("Radiotherapy", ["Yes", "No"])
        hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])

    # Buttons
    c1, c2 = st.columns([1, 1])
    with c1:
        reset = st.form_submit_button("🔄 Reset")
    with c2:
        submit = st.form_submit_button("🔍 Predict")

# Close Container
st.markdown("</div>", unsafe_allow_html=True)

# Reset Logic
if reset:
    st.experimental_rerun()

# Validation and Submission Logic
if submit:
    # Check if all fields are filled (which is implicitly done by default since all fields have default values)
    if age and menopausal_status and tumor_stage and lymph_nodes_examined is not None and er_status and pr_status and her2_status and chemotherapy and surgery and radiotherapy and hormone_therapy:
        st.success("Prediction functionality coming soon...")
    else:
        st.error("Please fill in all fields before predicting.")

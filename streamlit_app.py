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

/* Input styling */
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

# --- FORM START ---
with st.form("input_form"):
    st.markdown("<div class='section-title'>🧬 Clinical Data</div>", unsafe_allow_html=True)
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

    st.markdown("<div class='section-title'>💊 Treatment Data</div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        chemotherapy = st.selectbox("Chemotherapy", ["Yes", "No"])
        surgery = st.selectbox("Surgery Type", ["Breast-conserving", "Mastectomy"])
    with col4:
        radiotherapy = st.selectbox("Radiotherapy", ["Yes", "No"])
        hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])

    submitted = st.form_submit_button("Submit")  # Dummy submit to trigger form validation

# --- Custom Buttons (Outside Form) ---
colA, colB = st.columns(2)
with colA:
    if st.button("🔄 RESET"):
        st.experimental_rerun()
with colB:
    predict_btn = st.button("🔍 PREDICT")

# Custom CSS for the buttons (override Streamlit theme completely)
st.markdown("""
<style>
div.stButton > button {
    background-color: #ad1457 !important;
    color: white !important;
    font-weight: bold !important;
    text-transform: uppercase !important;
    border: none;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    margin-top: 0.5rem !important;
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

# --- Prediction Logic ---
if predict_btn:
    st.success("Prediction functionality coming soon...")

# --- Close Container ---
st.markdown("</div>", unsafe_allow_html=True)

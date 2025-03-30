import streamlit as st

# Set wide layout
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# --- Pink Theme Override ---
st.markdown("""
<style>
/* Body background */
body {
    background-color: #ffe3ec !important;
}

/* Card-like container */
.stApp {
    background-color: #ffe3ec !important;
}

/* Title */
h1 {
    text-align: center;
    color: #ad1457;
}

/* Section titles */
.section-title {
    font-size: 20px;
    font-weight: bold;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
    color: #ad1457;
}

/* Input fields */
div[data-baseweb="select"] > div,
input[type="number"] {
    background-color: #ffc3d9 !important;
    border-radius: 10px !important;
    padding: 8px !important;
    color: black !important;
}

/* Button styling */
button[aria-label="🔍 Predict"],
button[aria-label="🔄 Reset"] {
    background-color: #ad1457 !important;
    color: white !important;
    font-weight: bold !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    margin-top: 1rem !important;
    cursor: pointer !important;
}

/* Hover cursor on all fields */
input, select, textarea {
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

# --- Reset logic using query_params ---
if "reset" in st.query_params:
    st.query_params.clear()
    st.rerun()

# --- UI Header ---
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown("<h1>🎀 Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Fill in the details below to generate predictions and insights.</p>", unsafe_allow_html=True)

# --- FORM START ---
with st.form("input_form", clear_on_submit=False):
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

    # Custom Button Row
    colA, colB = st.columns(2)
    with colA:
        reset = st.form_submit_button("🔄 Reset")
    with colB:
        predict = st.form_submit_button("🔍 Predict")

# --- Button Actions ---
if reset:
    st.query_params["reset"] = "true"
    st.rerun()

if predict:
    st.success("Prediction functionality coming soon...")

st.markdown("</div>", unsafe_allow_html=True)

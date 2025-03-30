import streamlit as st

# Set page config
st.set_page_config(page_title="Breast Cancer Survival UI", layout="centered")

# ---- Custom CSS ----
st.markdown("""
    <style>
        .main-container {
            background-color: #f9b3c2;
            padding: 2rem;
            border-radius: 30px;
            width: 100%;
            max-width: 1000px;
            margin: auto;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .section-title {
            font-size: 22px;
            font-weight: bold;
            color: #C2185B;
            margin-bottom: 10px;
        }
        .stButton>button {
            width: 100%;
            border-radius: 12px;
        }
        .stTextInput>div>input, .stSelectbox>div>div, .stNumberInput>div>input {
            background-color: #ffc3d9 !important;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Page Layout ----
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Title and subtitle
    st.markdown("<h1 style='text-align: center;'>🎀 Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Fill in the details below to generate predictions and insights.</p>", unsafe_allow_html=True)

    # ---- Form Start ----
    with st.form("survival_form"):

        # Clinical Data Section
        st.markdown("### 🧬 Clinical Data")
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

        # Treatment Data Section
        st.markdown("### 💊 Treatment Data")
        col3, col4 = st.columns(2)
        with col3:
            chemotherapy = st.selectbox("Chemotherapy", ["Yes", "No"])
            surgery = st.selectbox("Surgery Type", ["Breast-conserving", "Mastectomy"])
        with col4:
            radiotherapy = st.selectbox("Radiotherapy", ["Yes", "No"])
            hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])

        # Buttons
        col5, col6 = st.columns(2)
        with col5:
            reset = st.form_submit_button("🔄 Reset")
        with col6:
            submit = st.form_submit_button("🔍 Predict")

        # Reset logic
        if reset:
            st.experimental_rerun()

        # Prediction simulation
        if submit:
            st.success("✅ Prediction done! (Simulated output here...)")

    # ---- Close pink container ----
    st.markdown('</div>', unsafe_allow_html=True)

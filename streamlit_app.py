import streamlit as st

st.set_page_config(page_title="Breast Cancer UI", layout="wide")

# --- CSS Styling ---
st.markdown("""
<style>
body {
    background-color: white;
}

.outer-container {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 2vh 1vw;
}

.inner-box {
    background-color: #f9b3c2;
    width: 90%;
    min-height: 92vh;
    padding: 3rem;
    border-radius: 30px;
    box-shadow: 0 0 15px rgba(0,0,0,0.1);
}

h1 {
    text-align: center;
    color: #C2185B;
    font-size: 2.5rem;
    margin-bottom: 2rem;
}
</style>

<div class="outer-container">
    <div class="inner-box">
        <h1>🎀 Breast Cancer Survival Prediction Interface</h1>
        <p style="text-align:center;">Fill in the details below to generate predictions and insights.</p>
""", unsafe_allow_html=True)

# ---- Your Streamlit UI content goes here INSIDE the pink container ----
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

st.markdown('<br>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    chemo = st.selectbox("Chemotherapy", ["Yes", "No"])
    surgery = st.selectbox("Surgery Type", ["Breast-conserving", "Mastectomy"])
with col4:
    radio = st.selectbox("Radiotherapy", ["Yes", "No"])
    hormone = st.selectbox("Hormone Therapy", ["Yes", "No"])

st.markdown('<br>', unsafe_allow_html=True)
predict, reset = st.columns(2)
with predict:
    st.button("🔍 Predict")
with reset:
    st.button("🔄 Reset")

# ---- Close pink container and wrapper ----
st.markdown("""
    </div>
</div>
""", unsafe_allow_html=True)

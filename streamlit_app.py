import streamlit as st

st.set_page_config(page_title="Breast Cancer UI", layout="wide")

# ---- White background with centered pink UI box ----
st.markdown("""
<style>
body {
    background-color: white;
}

.outer-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 40px;
}

.inner-box {
    background-color: #f9b3c2;
    width: 80%;
    border-radius: 30px;
    padding: 50px;
    box-shadow: 0 0 15px rgba(0, 0, 0, 0.05);
    min-height: 90vh;
}
</style>
<div class="outer-container">
    <div class="inner-box">
""", unsafe_allow_html=True)

# ---- Title Goes Here ----
st.markdown("<h1 style='text-align:center; color:#C2185B;'>🎀 Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)

# ---- End Pink UI Box ----
st.markdown("</div></div>", unsafe_allow_html=True)

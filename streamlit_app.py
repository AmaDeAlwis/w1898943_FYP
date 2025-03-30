import streamlit as st

# Set page config
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# 🎨 Custom CSS
st.markdown("""
<style>
body {
    background-color: #f9b3c2;
}
.white-wrapper {
    background-color: white;
    margin: 2rem auto;
    padding: 3rem;
    border-radius: 30px;
    width: 90%;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}
h1 {
    text-align: center;
    color: #C2185B;
}
</style>
""", unsafe_allow_html=True)

# ✅ White wrapper with title inside
st.markdown('<div class="white-wrapper">', unsafe_allow_html=True)

st.markdown("<h1>🎀 Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)
st.markdown("#### Fill in the details below to generate predictions and insights.")

# END of white wrapper
st.markdown("</div>", unsafe_allow_html=True)

import streamlit as st

# Page setup
st.set_page_config(page_title="Breast Cancer UI", layout="wide")

# Custom CSS for centered pink card with white border
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
    text-align: center;
}
h1 {
    color: #C2185B;
}
</style>
""", unsafe_allow_html=True)

# Layout begins
st.markdown('<div class="page-wrapper"><div class="bordered-container">', unsafe_allow_html=True)

# Title inside the pink box
st.markdown("<h1>🎀 Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)

# Layout ends
st.markdown('</div></div>', unsafe_allow_html=True)

import streamlit as st

# Page setup
st.set_page_config(page_title="Breast Cancer UI", layout="wide")

# Fix CSS to simulate white border by using a background wrapper div
st.markdown("""
<style>
html, body {
    background-color: white;
}

.main-container {
    background-color: #f9b3c2;
    border-radius: 30px;
    padding: 60px;
    width: 98vw;             /* Full width of the viewport */
    min-height: 95vh;        /* Full height of the viewport */
    margin: 20px auto;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# Main pink box (centered with margin)
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Title inside the pink container
st.markdown("<h1 style='text-align:center; color:#C2185B;'>🎀 Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)

# End the pink container
st.markdown("</div>", unsafe_allow_html=True)

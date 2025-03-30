import streamlit as st

# Set page config
st.set_page_config(page_title="Breast Cancer UI", layout="wide")

# Custom CSS to match the layout
st.markdown("""
<style>
/* Full white background */
body {
    background-color: white;
}

/* Centered pink card with rounded corners */
.pink-wrapper {
    background-color: #f9b3c2;
    border-radius: 35px;
    margin: 40px auto;
    padding: 50px;
    width: 85%;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# Pink wrapper start
st.markdown('<div class="pink-wrapper">', unsafe_allow_html=True)

# (Leave empty or insert title/inputs here later)
# st.markdown("<h1 style='text-align:center;'>🎀 Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)

# Pink wrapper end
st.markdown('</div>', unsafe_allow_html=True)

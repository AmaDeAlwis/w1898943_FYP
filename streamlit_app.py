import streamlit as st

st.set_page_config(page_title="Breast Cancer UI", layout="wide")

# ---- Custom CSS for fixed white border and large pink area ----
st.markdown("""
<style>
body {
    background-color: white;
}

.outer-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 4vh 2vw;
}

.inner-box {
    background-color: #f9b3c2;
    width: 100%;
    max-width: 1100px;
    min-height: 90vh;
    padding: 50px;
    border-radius: 25px;
    box-shadow: 0 0 10px rgba(0,0,0,0.08);
}

h1 {
    text-align: center;
    color: #C2185B;
    font-size: 2.4rem;
    margin-bottom: 30px;
}
</style>

<div class="outer-container">
    <div class="inner-box">
""", unsafe_allow_html=True)

# ---- All content INSIDE the pink container ----
st.markdown("<h1>🎀 Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)

# Example placeholder
st.write("🩺 All inputs and content will appear here inside the pink area...")

# ---- Close pink UI box and container ----
st.markdown("</div></div>", unsafe_allow_html=True)

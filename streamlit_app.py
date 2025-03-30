import streamlit as st

st.set_page_config(page_title="Breast Cancer UI", layout="wide")

# ---- Custom CSS for white border and large pink UI ----
st.markdown("""
<style>
body {
    background-color: white;
}

.outer-container {
    display: flex;
    justify-content: center;
    padding: 2vh 2vw;
}

.inner-box {
    background-color: #f9b3c2;
    width: 96vw;              /* Wider pink box */
    min-height: 92vh;
    padding: 50px;
    border-radius: 25px;
    box-shadow: 0 0 15px rgba(0,0,0,0.05);
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

# ---- Title inside pink box ----
st.markdown("<h1>🎀 Breast Cancer Survival Prediction Interface</h1>", unsafe_allow_html=True)

# ---- You can now add your inputs/forms here! ----
st.write("🩺 Form inputs will be placed here...")

# ---- Close pink UI box ----
st.markdown("</div></div>", unsafe_allow_html=True)

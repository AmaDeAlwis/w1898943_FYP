import streamlit as st
import pandas as pd
import numpy as np

#Page config
st.set_page_config(page_title="Breast Cancer Survival Prediction", layout="wide")

#Main title
st.title("🎀 Breast Cancer Survival Probability Prediction")

#Subheading or introduction
st.markdown("""
Welcome to the Breast Cancer Survival Prediction App.  
Enter patient data to estimate survival probability based on medical features.
""")

#Sidebar input section
st.sidebar.header("📝 Patient Information")
age = st.sidebar.slider("Age", 20, 90, 50)
tumor_size = st.sidebar.slider("Tumor Size (mm)", 0, 100, 20)
node_status = st.sidebar.selectbox("Node Status", ["Negative", "Positive"])
er_status = st.sidebar.radio("ER Status", ["Positive", "Negative"])
treatment = st.sidebar.selectbox("Treatment Type", ["Chemotherapy", "Hormone Therapy", "Radiotherapy", "No Treatment"])

#Convert input into numerical format (as needed for model)
input_data = {
    "Age": age,
    "Tumor_Size": tumor_size,
    "Node_Status": 1 if node_status == "Positive" else 0,
    "ER_Status": 1 if er_status == "Positive" else 0,
    "Treatment_Chemo": 1 if treatment == "Chemotherapy" else 0,
    "Treatment_Hormone": 1 if treatment == "Hormone Therapy" else 0,
    "Treatment_Radio": 1 if treatment == "Radiotherapy" else 0
}

#Display inputs as a table
st.subheader("🔍 Input Summary")
st.write(pd.DataFrame([input_data]))

#Placeholder for prediction (replace with actual model code)
if st.button("🔮 Predict Survival Probability"):
    #Dummy prediction (replace with model.predict_proba or similar)
    probability = np.random.uniform(0.6, 0.95)  # Simulated result
    st.success(f"Estimated 5-Year Survival Probability: **{probability:.2%}**")

    #Visualization
    st.subheader(" Survival Probability Gauge")
    st.progress(probability)

#Footer
st.markdown("---")
st.caption("Built with Streamlit")


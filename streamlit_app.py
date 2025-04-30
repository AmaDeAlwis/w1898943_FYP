import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pymongo import MongoClient
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from lifelines import CoxPHFitter

# --- Keys for resetting ---
reset_keys = ["patient_id", "age", "nodes", "meno", "stage", "her2", "er", "pr", "chemo", "surgery", "radio", "hormone"]

# --- Default values for form inputs ---
if st.session_state.get("reset_triggered"):
    default_values = {k: "" for k in reset_keys}
    st.session_state.pop("reset_triggered")
else:
    default_values = {k: st.session_state.get(k, "") for k in reset_keys}

# --- Load model and scaler ---
cox_model = joblib.load(".streamlit/cox_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- MongoDB connection ---
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# --- Page setup ---
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")
st.markdown("""
<style>
.custom-title {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    color: #ad1457;
    margin-bottom: 2rem;
}
.result-heading {
    font-size: 22px;
    color: #ad1457;
    font-weight: bold;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
.white-box {
    background-color: white;
    padding: 1.5rem;
    border-radius: 10px;
    margin-top: 1rem;
    margin-bottom: 2rem;
    box-shadow: 0 0 5px rgba(0,0,0,0.1);
}
.stButton button {
    background-color: #ad1457 !important;
    color: white !important;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-title">Breast Cancer Survival Prediction</div>', unsafe_allow_html=True)


# --- Patient ID ---
patient_id = st.text_input("Patient ID (Required)", value=default_values["patient_id"], key="patient_id")
if patient_id:
    prev = list(collection.find({"patient_id": patient_id}))
    if prev:
        with st.expander("Previous Predictions"):
            for r in prev:
                st.write(f"{r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ➔ 5yr: {r['survival_5yr']:.2f}, 10yr: {r['survival_10yr']:.2f}")

# --- Inputs ---
st.markdown("<div class='section-title'>Clinical Information</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age", value=default_values["age"], key="age")
    age_valid = True
    if age:
        try:
            age_val = float(age)
            if age_val < 20:
                age_valid = False
                st.markdown("<p style='color: #d6336c;'> Age must be at least 20.</p>", unsafe_allow_html=True)
        except ValueError:
            age_valid = False
            st.markdown("<p style='color: #d6336c;'> Age must be a valid number.</p>", unsafe_allow_html=True)

    lymph_nodes = st.text_input("Lymph Nodes Examined", value=default_values["nodes"], key="nodes")
    nodes_valid = True
    if lymph_nodes:
        try:
            nodes_val = float(lymph_nodes)
            if nodes_val < 0:
                nodes_valid = False
                st.markdown("<p style='color: #d6336c;'> Lymph Nodes must be 0 or greater.</p>", unsafe_allow_html=True)
        except ValueError:
            nodes_valid = False
            st.markdown("<p style='color: #d6336c;'> Lymph Nodes must be a valid number.</p>", unsafe_allow_html=True)

    meno_opts = ["", "Pre-menopausal", "Post-menopausal"]
    menopausal_status = st.selectbox("Menopausal Status", meno_opts,
                                     index=meno_opts.index(default_values["meno"]) if default_values["meno"] in meno_opts else 0,
                                     key="meno")
    stage_opts = ["", 1, 2, 3, 4]
    tumor_stage = st.selectbox("Tumor Stage", stage_opts,
                               index=stage_opts.index(int(default_values["stage"])) if str(default_values["stage"]).isdigit() and int(default_values["stage"]) in stage_opts else 0,
                               key="stage")

with col2:
    her2_opts = ["", "Neutral", "Loss", "Gain", "Undef"]
    her2 = st.selectbox("HER2 Status", her2_opts,
                        index=her2_opts.index(default_values["her2"]) if default_values["her2"] in her2_opts else 0,
                        key="her2")
    er_opts = ["", "Positive", "Negative"]
    er = st.selectbox("ER Status", er_opts,
                      index=er_opts.index(default_values["er"]) if default_values["er"] in er_opts else 0,
                      key="er")
    pr_opts = ["", "Positive", "Negative"]
    pr = st.selectbox("PR Status", pr_opts,
                      index=pr_opts.index(default_values["pr"]) if default_values["pr"] in pr_opts else 0,
                      key="pr")

st.markdown("<div class='section-title'>Treatment Information</div>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    chemo_opts = ["", "Yes", "No"]
    chemo = st.selectbox("Chemotherapy", chemo_opts,
                         index=chemo_opts.index(default_values["chemo"]) if default_values["chemo"] in chemo_opts else 0,
                         key="chemo")
    surgery_opts = ["", "Breast-conserving", "Mastectomy"]
    surgery = st.selectbox("Surgery Type", surgery_opts,
                           index=surgery_opts.index(default_values["surgery"]) if default_values["surgery"] in surgery_opts else 0,
                           key="surgery")
with col4:
    radio_opts = ["", "Yes", "No"]
    radio = st.selectbox("Radiotherapy", radio_opts,
                         index=radio_opts.index(default_values["radio"]) if default_values["radio"] in radio_opts else 0,
                         key="radio")
    hormone_opts = ["", "Yes", "No"]
    hormone = st.selectbox("Hormone Therapy", hormone_opts,
                           index=hormone_opts.index(default_values["hormone"]) if default_values["hormone"] in hormone_opts else 0,
                           key="hormone")

# --- Buttons ---
predict = False
col_b1, col_b2 = st.columns(2)
with col_b1:
    if st.button("RESET"):
        st.session_state.clear()
        st.session_state["reset_triggered"] = True
        st.rerun()
with col_b2:
    predict = st.button("PREDICT")

if predict and patient_id:
    if "" in [age, lymph_nodes, menopausal_status, er, pr, her2, chemo, radio, hormone, surgery, tumor_stage] or not age_valid or not nodes_valid:
        st.error("Please fill out all fields correctly before predicting.")
    else:
        menopausal = 1 if menopausal_status == "Post-menopausal" else 0
        er = 1 if er == "Positive" else 0
        pr = 1 if pr == "Positive" else 0
        her2_vals = [0, 0, 0, 0]
        her2_opts = ["Gain", "Loss", "Neutral", "Undef"]
        if her2 in her2_opts:
            her2_vals[her2_opts.index(her2)] = 1
        chemo = 1 if chemo == "Yes" else 0
        radio = 1 if radio == "Yes" else 0
        hormone = 1 if hormone == "Yes" else 0
        surgery_conserve = 1 if surgery == "Breast-conserving" else 0
        surgery_mastectomy = 1 if surgery == "Mastectomy" else 0

        features = np.array([
            float(age), chemo, er, hormone, menopausal, float(lymph_nodes), pr, radio, int(tumor_stage),
            surgery_conserve, surgery_mastectomy, *her2_vals
        ]).reshape(1, -1)

        features_scaled = scaler.transform(features)
        df_input = pd.DataFrame(features_scaled, columns=cox_model.params_.index)

        surv_func = cox_model.predict_survival_function(df_input)
        times = surv_func.index.values
        surv_5yr = np.interp(60, times, surv_func.values.flatten())
        surv_10yr = np.interp(120, times, surv_func.values.flatten())

        st.session_state["surv_5yr"] = surv_5yr
        st.session_state["surv_10yr"] = surv_10yr
        st.session_state["surv_times"] = times
        st.session_state["surv_func_values"] = surv_func.values.flatten()
        st.session_state["saved_patient_id"] = patient_id

        collection.insert_one({
            "patient_id": patient_id,
            "timestamp": pd.Timestamp.now(),
            "survival_5yr": float(surv_5yr),
            "survival_10yr": float(surv_10yr)
        })

        st.success(" Prediction complete and saved!")

# --- Render Results if Available ---
if "surv_5yr" in st.session_state:
    surv_5yr = st.session_state["surv_5yr"]
    surv_10yr = st.session_state["surv_10yr"]
    times = st.session_state["surv_times"]
    surv_func_values = st.session_state["surv_func_values"]
    patient_id = st.session_state["saved_patient_id"]

    with st.container():
        st.markdown(f"""
            <div class='white-box'>
                <div class='result-heading'>Survival Predictions</div>
                <p><strong>5-Year Survival Probability:</strong> {surv_5yr:.2f} ({surv_5yr * 100:.0f}%)</p>
                <p><strong>10-Year Survival Probability:</strong> {surv_10yr:.2f} ({surv_10yr * 100:.0f}%)</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Results Overview</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        fig, ax = plt.subplots()
        ax.bar(["5-Year", "10-Year"], [surv_5yr, surv_10yr], color="#FF69B4")
        for i, v in enumerate([surv_5yr, surv_10yr]):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
        ax.set_ylim(0, 1)
        st.pyplot(fig)

    with c2:
        if surv_5yr < 0.5:
            st.error("Low Survival Chance")
            st.info("Patient shows low probability. Consider aggressive treatment planning.")
        elif surv_5yr < 0.75:
            st.warning("Moderate Survival Chance")
            st.info("Patient is at moderate risk. Monitor closely and adjust treatment accordingly.")
        else:
            st.success("High Survival Chance")
            st.info("Patient has a favorable survival outlook. Continue regular monitoring.")
 

    with c3:
        fig2, ax2 = plt.subplots()
        ax2.plot(times, surv_func_values, color="#c2185b")
        ax2.set_title("Survival Curve")
        ax2.set_xlabel("Time (Months)")
        ax2.set_ylabel("Survival Probability")
        st.pyplot(fig2)

        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, 770, " Breast Cancer Survival Prediction Report")
        
        c.setFont("Helvetica", 12)
        c.drawString(100, 740, f"Patient ID: {st.session_state['saved_patient_id']}")
        c.drawString(100, 720, f"5-Year Survival Probability: {surv_5yr:.2f} ({surv_5yr * 100:.0f}%)")
        c.drawString(100, 700, f"10-Year Survival Probability: {surv_10yr:.2f} ({surv_10yr * 100:.0f}%)")
        
        # Add risk tag and recommendation
        if surv_5yr < 0.5:
            risk = "High Risk"
            recommendation = "Consider aggressive treatment planning."
        elif surv_5yr < 0.75:
            risk = "Moderate Risk"
            recommendation = "Monitor closely and adjust treatment accordingly."
        else:
            risk = "Low Risk"
            recommendation = "Favorable outlook. Continue regular follow-up."
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, 670, f"Risk Category: {risk}")
        c.setFont("Helvetica", 12)
        c.drawString(100, 650, f"Recommendation: {recommendation}")
        
        c.save()
        pdf_buffer.seek(0)


    st.markdown("<p style='color:#ad1457; font-weight:bold; margin-top:1rem;'> Download your report:</p>", unsafe_allow_html=True)
    st.download_button(
        label="Download Report",
        data=pdf_buffer,
        file_name=f"Survival_Report_{st.session_state['saved_patient_id']}.pdf",
        mime="application/pdf",
        help="Download a summary of the prediction"
    )

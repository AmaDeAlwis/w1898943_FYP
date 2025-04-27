import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import joblib
from pymongo import MongoClient
import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from gcn_model_class import SurvivalGNN

# --- Streamlit Config ---
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# --- Load Model & Scaler ---
gcn_model = SurvivalGNN(in_channels=15, out_channels_time=1, out_channels_event=1)
gcn_model.load_state_dict(torch.load(".streamlit/gcn_model.pt", map_location=torch.device("cpu")))
gcn_model.eval()
scaler = joblib.load("scaler.pkl")

# --- MongoDB Connection ---
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["breast_cancer_survival"]
collection = db["patient_records"]

# --- Initialize patient_id if not set ---
if "patient_id" not in st.session_state:
    st.session_state["patient_id"] = ""

# --- Field Keys ---
field_keys = [
    "age", "menopausal_status", "tumor_stage", "lymph_nodes_examined",
    "er_status", "pr_status", "her2_status", "chemotherapy",
    "surgery", "radiotherapy", "hormone_therapy"
]

# --- Custom Styling ---
st.markdown("""
<style>
h1 {
    color: #ad1457;
    text-align: center;
    font-weight: bold;
}
.section-title {
    font-size: 22px;
    font-weight: bold;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
    color: #ad1457;
}
.stButton button {
    background-color: #ad1457 !important;
    color: white !important;
    border-radius: 10px;
    font-weight: bold;
}
.result-heading {
    font-size: 22px;
    color: #c2185b;
    margin-top: 2rem;
    font-weight: bold;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1> Breast Cancer Survival Prediction </h1>", unsafe_allow_html=True)

# --- Patient Information ---
st.markdown("<p class='section-title'>Patient Information</p>", unsafe_allow_html=True)
patient_id = st.text_input("Patient ID (Required)", value=st.session_state["patient_id"], key="patient_id")

# --- Show Previous Predictions ---
if patient_id:
    previous_records = list(collection.find({"patient_id": patient_id}))
    if previous_records:
        with st.expander("View Previous Predictions for this Patient ID"):
            for record in previous_records:
                st.write(f"Date: {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"🔹 5-Year Survival: {record['survival_5yr']:.2f}")
                st.write(f"🔹 10-Year Survival: {record['survival_10yr']:.2f}")
                st.markdown("---")

# --- Clinical Information ---
st.markdown("<p class='section-title'>Clinical Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age", value=st.session_state.get("age", ""), key="age")
    if st.session_state.get("age", ""):
        if not st.session_state.age.isdigit():
            st.warning("Age must be a number.")
        elif int(st.session_state.age) < 20:
            st.warning("Age must be at least 20.")

    menopausal_status = st.selectbox("Menopausal Status", ["", "Pre-menopausal", "Post-menopausal"],
                                     index=0 if "menopausal_status" not in st.session_state else
                                     ["", "Pre-menopausal", "Post-menopausal"].index(st.session_state["menopausal_status"]),
                                     key="menopausal_status")

    tumor_stage = st.selectbox("Tumor Stage", ["", 1, 2, 3, 4],
                               index=0 if "tumor_stage" not in st.session_state else
                               ["", 1, 2, 3, 4].index(st.session_state["tumor_stage"]),
                               key="tumor_stage")

    lymph_nodes_examined = st.text_input("Lymph Nodes Examined", value=st.session_state.get("lymph_nodes_examined", ""), key="lymph_nodes_examined")
    if st.session_state.get("lymph_nodes_examined", ""):
        if not st.session_state.lymph_nodes_examined.isdigit():
            st.warning("Lymph Nodes must be a number.")
        elif int(st.session_state.lymph_nodes_examined) < 0:
            st.warning("Lymph Nodes must be 0 or more.")

with col2:
    er_status = st.selectbox("ER Status", ["", "Positive", "Negative"],
                             index=0 if "er_status" not in st.session_state else
                             ["", "Positive", "Negative"].index(st.session_state["er_status"]),
                             key="er_status")

    pr_status = st.selectbox("PR Status", ["", "Positive", "Negative"],
                             index=0 if "pr_status" not in st.session_state else
                             ["", "Positive", "Negative"].index(st.session_state["pr_status"]),
                             key="pr_status")

    her2_status = st.selectbox("HER2 Status", ["", "Neutral", "Loss", "Gain", "Undef"],
                               index=0 if "her2_status" not in st.session_state else
                               ["", "Neutral", "Loss", "Gain", "Undef"].index(st.session_state["her2_status"]),
                               key="her2_status")

# --- Treatment Information ---
st.markdown("<p class='section-title'>Treatment Information</p>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    chemotherapy = st.selectbox("Chemotherapy", ["", "Yes", "No"],
                                index=0 if "chemotherapy" not in st.session_state else
                                ["", "Yes", "No"].index(st.session_state["chemotherapy"]),
                                key="chemotherapy")

    surgery = st.selectbox("Surgery Type", ["", "Breast-conserving", "Mastectomy"],
                           index=0 if "surgery" not in st.session_state else
                           ["", "Breast-conserving", "Mastectomy"].index(st.session_state["surgery"]),
                           key="surgery")

with col4:
    radiotherapy = st.selectbox("Radiotherapy", ["", "Yes", "No"],
                                index=0 if "radiotherapy" not in st.session_state else
                                ["", "Yes", "No"].index(st.session_state["radiotherapy"]),
                                key="radiotherapy")

    hormone_therapy = st.selectbox("Hormone Therapy", ["", "Yes", "No"],
                                   index=0 if "hormone_therapy" not in st.session_state else
                                   ["", "Yes", "No"].index(st.session_state["hormone_therapy"]),
                                   key="hormone_therapy")

# --- Buttons ---
left, right = st.columns(2)
with left:
    if st.button("RESET"):
        for k in list(st.session_state.keys()):
            if k in field_keys + ["patient_id"]:
                del st.session_state[k]
        st.rerun()

with right:
    predict_clicked = st.button("PREDICT")

# --- Prediction Logic ---
if predict_clicked:
    required_fields = [st.session_state.get(k, "") for k in field_keys]

    if not patient_id:
        st.warning("Please enter a Patient ID to save the record")
    elif "" in required_fields:
        st.warning("Please fill all required fields")
    else:
        age = int(st.session_state.age)
        lymph_nodes_examined = int(st.session_state.lymph_nodes_examined)
        menopausal_status = 1 if st.session_state.menopausal_status == "Post-menopausal" else 0
        er_status = 1 if st.session_state.er_status == "Positive" else 0
        pr_status = 1 if st.session_state.pr_status == "Positive" else 0
        her2_val = st.session_state.her2_status
        her2_neutral = 1 if her2_val == "Neutral" else 0
        her2_loss = 1 if her2_val == "Loss" else 0
        her2_gain = 1 if her2_val == "Gain" else 0
        her2_undef = 1 if her2_val == "Undef" else 0
        chemotherapy = 1 if st.session_state.chemotherapy == "Yes" else 0
        radiotherapy = 1 if st.session_state.radiotherapy == "Yes" else 0
        hormone_therapy = 1 if st.session_state.hormone_therapy == "Yes" else 0
        surgery_conserving = 1 if st.session_state.surgery == "Breast-conserving" else 0
        surgery_mastectomy = 1 if st.session_state.surgery == "Mastectomy" else 0
        tumor_stage = int(st.session_state.tumor_stage)

        input_features = np.array([
            age, chemotherapy, er_status, hormone_therapy, menopausal_status,
            lymph_nodes_examined, pr_status, radiotherapy, tumor_stage,
            surgery_conserving, surgery_mastectomy, her2_gain,
            her2_loss, her2_neutral, her2_undef
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_features)
        x_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        graph_data = Data(x=x_tensor, edge_index=edge_index)

        with torch.no_grad():
            time_output, event_output = gcn_model(graph_data)
            survival_5yr = torch.sigmoid(time_output[0]).item()
            survival_10yr = torch.sigmoid(event_output[0]).item()

        st.success("✅ Patient record successfully saved!")

        # --- Results Overview Heading ---
        st.markdown("<p class='result-heading'>Results Overview</p>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            fig_bar, ax_bar = plt.subplots(figsize=(3, 3))
            bars = ax_bar.bar(["5-Year", "10-Year"], [survival_5yr, survival_10yr], color="#FF69B4")
            ax_bar.set_ylim(0, 1)
            ax_bar.set_ylabel("Probability", fontsize=10)
            ax_bar.set_title("Survival Probability", fontsize=12, fontweight="bold", pad=10)
            for bar, value in zip(bars, [survival_5yr, survival_10yr]):
                ax_bar.text(bar.get_x() + bar.get_width()/2, value + 0.02, f"{value:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            st.pyplot(fig_bar)

        with col2:
            risk_text = '🔴 Low Survival Chance' if survival_5yr < 0.6 else '🟡 Moderate Survival Chance' if survival_5yr < 0.8 else '🟢 High Survival Chance'
            recommendation_text = 'Consider aggressive treatment planning.' if survival_5yr < 0.6 else 'Consider more frequent follow-up.' if survival_5yr < 0.8 else 'Continue standard monitoring.'

            st.markdown(
                f"""
                <div style='background-color: #ffffff; padding: 2rem; border-radius: 20px; height: 442px; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
                    <div style='color: red; font-weight: bold; font-size: 20px; margin-bottom: 1rem;'>{risk_text}</div>
                    <div style='color: #333366; font-size: 16px; text-align: center;'>{recommendation_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            fig_curve, ax_curve = plt.subplots(figsize=(3, 3))
            x_vals = np.array([0, 60, 120]) / 120
            y_vals = np.array([survival_5yr, (survival_5yr + survival_10yr)/2, survival_10yr])
            ax_curve.plot(x_vals, y_vals, color='#FF69B4', marker='o')
            ax_curve.set_ylim(0, 1)
            ax_curve.set_xlabel("Time", fontsize=10)
            ax_curve.set_ylabel("Survival Probability", fontsize=10)
            ax_curve.set_title("Estimated Survival Curve", fontsize=12, fontweight="bold")
            ax_curve.spines['top'].set_visible(False)
            ax_curve.spines['right'].set_visible(False)
            st.pyplot(fig_curve)

        # --- Download Button for PDF ---
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, height - 100, "Breast Cancer Survival Prediction Report")

        c.setFont("Helvetica", 12)
        c.drawString(100, height - 150, f"Patient ID: {patient_id}")
        c.drawString(100, height - 180, f"5-Year Survival Probability: {survival_5yr:.2f}")
        c.drawString(100, height - 210, f"10-Year Survival Probability: {survival_10yr:.2f}")
        c.drawString(100, height - 250, f"Risk Level: {risk_text}")
        c.drawString(100, height - 280, f"Recommendation: {recommendation_text}")

        c.save()
        pdf_buffer.seek(0)

        st.download_button(
            label="📄 Download Report as PDF",
            data=pdf_buffer,
            file_name=f"Survival_Report_{patient_id}.pdf",
            mime="application/pdf",
        )

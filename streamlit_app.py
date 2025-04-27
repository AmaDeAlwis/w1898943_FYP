# --- Middle - Risk & Recommendation ---
with col2:
    if survival_5yr > 0.8:
        tag = "High Survival Chance"
        reco = "🟢 Patient shows high probability. Continue standard monitoring."
    elif survival_5yr > 0.6:
        tag = "Moderate Survival Chance"
        reco = "🟡 Consider more frequent follow-up."
    else:
        tag = "Low Survival Chance"
        reco = "🔴 Consider aggressive treatment planning."

    st.markdown(
        f"""
        <div style='background-color: #ffe6e6; padding: 1rem; border-radius: 15px; height: 330px; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
            <h5 style='color:red;'>{tag}</h5>
            <p style='color: #444; font-size: 16px;'>{reco}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

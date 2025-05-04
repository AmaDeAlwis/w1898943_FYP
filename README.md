Breast Cancer Survival Probability Prediction System

Project Overview

This project aims to predict 5-year and 10-year survival probabilities for breast cancer patients using a combination of statistical, machine learning, and deep learning models. The system is built using the METABRIC dataset, which provides comprehensive clinical and treatment information, making it suitable for survival analysis. The objective is to aid clinicians in treatment planning and follow-up by offering interpretable, personalized survival predictions.

Models Used

Four models were fully implemented and evaluated for survival prediction:

Cox Proportional Hazards Model (CoxPH)

Aalen’s Additive Hazard Model

Random Survival Forest (RSF)

Graph Convolutional Network (GCN)

The CoxPH model was selected for deployment based on its strong performance and clinical interpretability. XGBoost was used exclusively for feature importance interpretation.

Methodology

The METABRIC dataset underwent extensive exploratory data analysis (EDA) to understand feature distributions and identify clinically relevant variables.
Following this, a detailed data preprocessing pipeline was applied, including missing value handling, encoding, normalization, and feature selection.
The models were trained on the preprocessed data, and cross-validation was conducted where applicable.
Performance was evaluated using survival-specific metrics such as Concordance Index, Integrated Brier Score, and Time-dependent AUC.
Calibration plots were also used to assess model reliability.
The final model (CoxPH ) with highest perfromance integrated into a Streamlit-based user interface, enabling users to input clinical and treatment data and receive personalized survival predictions.

User Interface

A user-friendly Streamlit interface was developed with a pink/white theme, aligning with the breast cancer awareness theme.
Users can enter clinical and treatment-related features grouped into dedicated sections. The system displays predicted 5-year and 10-year survival probabilities, visualizations, treatment recommendations, risk tags and downloadable PDF.
MongoDB Atlas is used as the backend to store and retrieve patient data, enabling follow-up tracking and future comparisons. The interface also supports resetting inputs and storing predictions securely.

Repository Information

Model training, evaluation, and testing were conducted using Google Colab, and the user interface was deployed via Streamlit Cloud.

Disclaimer

This project is developed solely for academic and research purposes. It is not intended for clinical use without further validation and regulatory approval.

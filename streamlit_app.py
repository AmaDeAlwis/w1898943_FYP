import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import matplotlib.pyplot as plt

# ✅ Set page config FIRST
st.set_page_config(page_title="Breast Cancer Survival UI", layout="wide")

# 🎨 Then apply custom CSS
def apply_custom_css():
    st.markdown("""
        <style>
        .stApp {
            background-color: #fff0f5;
        }
        section[data-testid="stSidebar"] {
            background-color: #ffe6f0;
        }
        h1, h2 {
            color: #d63384;
            text-align: center;
        }
        .stButton button {
            background-color: #ff69b4;
            color: white;
            border-radius: 10px;
        }
        .stButton button:hover {
            background-color: #ff85c1;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

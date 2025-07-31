import streamlit as st
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set up Streamlit
st.set_page_config(layout="centered", page_title="Churn Model Explanation")
st.title("🔍 Churn Prediction Model – SHAP Explanations")

# Load assets
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
tfidf = joblib.load("tfidf.pkl")

# Upload user data
uploaded_file = st.file_uploader("Upload CSV with customer data", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    structured_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    text_col = 'Feedback'

    df[text_col] = df[text_col].fillna("")
    df[structured_cols] = df[structured_cols].fillna(0)

    X_structured = scaler.transform(df[structured_cols])
    X_text = tfidf.transform(df[text_col]).toarray()
    X_final = np.hstack((X_structured, X_text))

    st.success("✅ Data processed and ready for explanation.")

    # SHAP summary plot
    st.subheader("📊 SHAP Summary Plot (Global Feature Importance)")
    explainer = shap.Explainer(model, X_final)
    shap_values = explainer(X_final)

    fig_summary = shap.summary_plot(shap_values, X_final, show=False)
    st.pyplot(bbox_inches='tight')

    # SHAP force plot (first prediction)
    st.subheader("🧠 SHAP Force Plot (First Prediction)")

    shap.initjs()
    st_shap = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        matplotlib=True,
        feature_names=[*structured_cols] + tfidf.get_feature_names_out().tolist()
    )
    st.pyplot(bbox_inches='tight')

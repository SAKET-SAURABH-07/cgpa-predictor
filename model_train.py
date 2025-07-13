import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_squared_error

# Load model and scaler
model = joblib.load("cgpa_best_model.pkl")
scaler = joblib.load("cgpa_scaler.pkl")

st.title("🎓 CGPA Predictor: Actual vs Predicted")

# Upload CSV for prediction
uploaded_file = st.file_uploader("Upload CSV file with student features (excluding next_sem_cgpa):", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'next_sem_cgpa' in df.columns:
        y_true = df['next_sem_cgpa']
        X = df.drop('next_sem_cgpa', axis=1)
    else:
        y_true = None
        X = df
    
    # Scale input features
    X_scaled = scaler.transform(X)

    # Make predictions
    predictions = model.predict(X_scaled)

    # Show table
    result_df = df.copy()
    result_df['Predicted_CGPA'] = predictions
    st.dataframe(result_df)

    # If actual CGPA exists, show graph
    if y_true is not None:
        r2 = r2_score(y_true, predictions)
        mse = mean_squared_error(y_true, predictions)
        st.subheader(f"📊 Model Performance")
        st.write(f"R² Score: `{r2:.4f}` | MSE: `{mse:.4f}`")

        # Plot Actual vs Predicted
        fig, ax = plt.subplots()
        ax.scatter(y_true, predictions, c='blue', alpha=0.6, label="Predicted")
        ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Perfect Prediction")
        ax.set_xlabel("Actual CGPA")
        ax.set_ylabel("Predicted CGPA")
        ax.set_title("Actual vs Predicted CGPA")
        ax.legend()
        st.pyplot(fig)

else:
    st.info("Upload a CSV file to start predicting.")


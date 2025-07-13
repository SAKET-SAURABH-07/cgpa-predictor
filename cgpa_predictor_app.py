import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load("cgpa_best_model.pkl")
scaler = joblib.load("cgpa_scaler.pkl")

st.title("🎓 CGPA Predictor (Slider Input + Prediction Plot)")

st.markdown("Adjust the sliders below to simulate a student's profile and predict the **next semester CGPA**.")

# Slider inputs
prev_cgpa = st.slider("Previous Semester CGPA", 4.0, 10.0, 7.0, step=0.1)
study_hours = st.slider("Average Study Hours per Day", 0.0, 10.0, 5.0, step=0.1)
attendance = st.slider("Attendance Percent", 0, 100, 75, step=1)
sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, step=0.1)
extracurricular = st.slider("Extracurricular Score", 0.0, 10.0, 5.0, step=0.1)
projects_done = st.slider("Number of Projects Done", 0, 10, 2, step=1)
internet_usage = st.slider("Internet Usage Hours per Day", 0.0, 24.0, 4.0, step=0.1)
stress_level = st.slider("Stress Level (1=Low, 10=High)", 1.0, 10.0, 5.0, step=0.1)

# Input vector
input_features = np.array([[prev_cgpa, study_hours, attendance, sleep_hours,
                            extracurricular, projects_done, internet_usage, stress_level]])

# Scale and predict
scaled_features = scaler.transform(input_features)
predicted_cgpa = model.predict(scaled_features)[0]

# Show prediction
st.success(f"📘 Predicted Next Semester CGPA: **{predicted_cgpa:.2f}**")

# ⚠️ Disclaimer (visible right after prediction)
st.warning("""
⚠️ **Disclaimer:** This prediction is generated using a machine learning model trained on limited data.  
It is **not 100% accurate** and should be treated as a helpful **estimate**, not a guaranteed result.
""")

# Show graph
st.subheader("📈 Your CGPA Prediction on a Line Graph")

# Create dummy actuals for context
dummy_x = list(range(20))
dummy_actuals = np.linspace(4.5, 9.5, 20)
dummy_predictions = dummy_actuals + np.random.normal(0, 0.2, 20)
dummy_predictions = np.clip(dummy_predictions, 4.0, 10.0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(dummy_x, dummy_predictions, label="Other Predicted CGPA", linestyle='--', color='blue')
ax.scatter([len(dummy_x)], [predicted_cgpa], color='red', s=100, label="Your Prediction")
ax.set_xlabel("Student Index")
ax.set_ylabel("Predicted CGPA")
ax.set_title("Your Prediction vs Others")
ax.legend()
ax.grid(True)

st.pyplot(fig)

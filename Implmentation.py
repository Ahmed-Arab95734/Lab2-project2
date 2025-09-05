import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title
st.title("Churn Prediction by Tenure")

# load saved model
model = joblib.load("GaussianNB.pkl")
Churn_labels = {0:"No", 1:"Yes"}


Age = st.sidebar.slider("Age (years)", 18, 100, 25)
Tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
Sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

input_data=np.array([[Age, Tenure, 1 if Sex=="Male" else 0]])

prediction = model.predict(input_data) [0]

prediction_proba = model.predict_proba(input_data) [0]

# show prediction
st.subheader("Prediction Result")
st.write(f"Predicted Churn: {Churn_labels[prediction]}")

#show prediction probabilities
st.subheader("Churn Prediction Probabilities")
st.write(F"No {prediction_proba[0] :.2%}")
st.write(F"Yes {prediction_proba[1]:.2%}")
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the models and scaler
logreg_model = joblib.load('logreg_model.pkl')
svm_model = joblib.load('svm_model.pkl')
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to make prediction
def predict_bankruptcy(model, industrial_risk, management_risk, financial_flexibility,
                       credibility, competitiveness, operating_risk):
    # Prepare input data for prediction
    input_data = pd.DataFrame([[industrial_risk, management_risk, financial_flexibility,
                                credibility, competitiveness, operating_risk]],
                              columns=['industrial_risk', 'management_risk', 'financial_flexibility',
                                       'credibility', 'competitiveness', 'operating_risk'])
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict using the selected model
    prediction = model.predict(input_data_scaled)
    
    return prediction[0]

# Streamlit UI
st.title("Business Bankruptcy Prediction")

st.write("""
This tool predicts the likelihood of a company going bankrupt based on several features.
Please input the following features:
""")

# Input fields
industrial_risk = st.selectbox("Industrial Risk", [0, 0.5, 1])
management_risk = st.selectbox("Management Risk", [0, 0.5, 1])
financial_flexibility = st.selectbox("Financial Flexibility", [0, 0.5, 1])
credibility = st.selectbox("Credibility", [0, 0.5, 1])
competitiveness = st.selectbox("Competitiveness", [0, 0.5, 1])
operating_risk = st.selectbox("Operating Risk", [0, 0.5, 1])

# Model selection
model_choice = st.selectbox("Select Model", ["Logistic Regression", "Support Vector Machine", "Random Forest"])

# Choose model based on selection
if model_choice == "Logistic Regression":
    selected_model = logreg_model
elif model_choice == "Support Vector Machine":
    selected_model = svm_model
else:
    selected_model = rf_model

# Prediction button
if st.button("Predict"):
    result = predict_bankruptcy(selected_model, industrial_risk, management_risk, financial_flexibility,
                                credibility, competitiveness, operating_risk)
    
    if result == 0:
        st.write("The company is **not likely to go bankrupt**.")
    else:
        st.write("The company is **likely to go bankrupt**.")
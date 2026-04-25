import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and encoder
model = joblib.load('fraud_model.pkl')
encoder = joblib.load('label_encoder.pkl')

st.title("🛡️ Mobile Transaction Fraud Detection")
st.write("Enter transaction details below to check for potential fraud.")

# Input fields based on the dataset columns
col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Step (Hours since start)", min_value=1, value=1)
    type_options = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    transaction_type = st.selectbox("Transaction Type", type_options)
    amount = st.number_input("Amount", min_value=0.0, value=1000.0)

with col2:
    oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, value=5000.0)
    newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=4000.0)
    oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, value=1000.0)

if st.button("Predict Fraud"):
    # 1. Encode the categorical 'type' just like in the notebook
    encoded_type = encoder.transform([transaction_type])[0]
    
    # 2. Prepare features in the exact order used during training
    features = np.array([[
        step, encoded_type, amount, oldbalanceOrg, 
        newbalanceOrig, oldbalanceDest, newbalanceDest
    ]])
    
    # 3. Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]
    
    # 4. Display Results
    if prediction[0] == 1:
        st.error(f"🚨 High Risk: This transaction is likely FRAUDULENT (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk: This transaction appears LEGITIMATE (Probability of fraud: {probability:.2%})")

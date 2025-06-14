import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.title('Credit Card Fraud Detection')
st.write('Enter transaction details to check if it is fraudulent.')

# Load a small sample for demo (for speed)
@st.cache_data
def load_data():
    df = pd.read_csv('Credit-Card-Fraud-Detection-master/creditcard.csv', nrows=10000)  # Use a small sample for demo
    return df

data = load_data()

# Features for input (excluding 'Time' and 'Class')
features = [col for col in data.columns if col not in ['Time', 'Class']]

# Sidebar for user input
st.sidebar.header('Transaction Input')
user_input = {}
for feat in features:
    min_val = float(data[feat].min())
    max_val = float(data[feat].max())
    mean_val = float(data[feat].mean())
    user_input[feat] = st.sidebar.slider(f'{feat}', min_val, max_val, mean_val)

# Prepare data for prediction
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest (unsupervised, for demo)
model = IsolationForest(contamination=0.001, random_state=42)
model.fit(X_scaled)

# Predict for user input
user_df = pd.DataFrame([user_input])
user_scaled = scaler.transform(user_df)
pred = model.predict(user_scaled)

if st.button('Check for Fraud'):
    if pred[0] == -1:
        st.error('⚠️ This transaction is predicted to be FRAUDULENT!')
    else:
        st.success('✅ This transaction is predicted to be NORMAL.')

st.markdown('---')
st.write('Note: This demo retrains a simple model each time. For production, use a saved, tuned model.') 
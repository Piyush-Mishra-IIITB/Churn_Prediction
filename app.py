import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Page Config
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

# Load model and scaler
model = load_model("churn_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Title Section
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# ============ Input Section ============

st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=600)
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10, value=3)
    gender = st.selectbox("Gender", ["Female", "Male"])

with col2:
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)
    num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    has_credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])

st.subheader("Location & Income")
geo = st.selectbox("Geography", ["France", "Germany", "Spain"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

st.markdown("---")

# ============ Preprocessing ============

# Convert categorical fields
has_credit_card = 1 if has_credit_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0
gender_male = 1 if gender == "Male" else 0

# One-hot encode Geography
geo_germany = 1 if geo == "Germany" else 0
geo_spain = 1 if geo == "Spain" else 0

# Arrange in correct input order
input_data = np.array([[ 
    credit_score,
    age,
    tenure,
    balance,
    num_products,
    has_credit_card,
    is_active_member,
    estimated_salary,
    geo_germany,
    geo_spain,
    gender_male
]])

# Scaling
input_scaled = scaler.transform(input_data)

# ============ Prediction ============

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict Churn", use_container_width=True):
    prediction = model.predict(input_scaled)[0][0]
    
    st.markdown("---")
    
    if prediction > 0.5:
        st.markdown(
            f"<h3 style='color: red;'>Customer Will Churn</h3>"
            f"<p>Churn Probability: <strong>{prediction:.2f}</strong></p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<h3 style='color: green;'>Customer Will NOT Churn</h3>"
            f"<p>Churn Probability: <strong>{prediction:.2f}</strong></p>",
            unsafe_allow_html=True
        )

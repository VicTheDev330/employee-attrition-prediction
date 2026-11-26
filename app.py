import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("employee_attrition_app.pkl")

st.title("Employee Attrition Prediction")

st.write(
    "Predict whether an employee may leave the company "
    "using HR information."
)

# User inputs
age = st.number_input(
    "Employee Age",
    min_value=18,
    max_value=65,
    value=30
)

income = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=20000,
    value=5000
)

satisfaction = st.selectbox(
    "Job Satisfaction",
    [1, 2, 3, 4]
)

overtime = st.selectbox(
    "OverTime",
    ["Yes", "No"]
)

# Prediction
if st.button("Predict Attrition"):

    input_data = pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": income,
        "JobSatisfaction": satisfaction,
        "OverTime": overtime
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("Employee may leave the company")
    else:
        st.success("Employee may stay in the company")

    st.write(
        "Attrition Probability:",
        f"{probability:.2%}"
    )

    st.caption(
        "This is a machine learning estimate, "
        "not a guarantee of employee behaviour."
    )
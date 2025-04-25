import streamlit as st
import pandas as pd
import joblib


# Load the trained pipeline from disk
pipeline = joblib.load("models/loan_pipeline.pkl")

st.title("Loan Approval Predictor")

with st.form("loan_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["male", "female"],format_func=lambda x: x.title())
    education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
    income = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
    emp_exp = st.number_input("Years of Employment", min_value=0, max_value=50, value=5)
    home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"],format_func=lambda x: x.title())
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=10000.0, step=500.0)
    loan_intent = st.selectbox("Loan Purpose", [
        "PERSONAL", "EDUCATION", "MEDICAL",
        "VENTURE", "HOME_IMPROVEMENT", "DEBT_CONSOLIDATION"
    ],format_func=lambda x: x.title())
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
    percent_income = st.number_input("Loan as % of Income", min_value=0.0, max_value=200.0, value=20.0)
    credit_history = st.number_input("Credit History (years)", min_value=0.0, max_value=50.0, value=5.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    previous_defaults = st.selectbox("Previous Defaults?", ["No", "Yes"])
    submit = st.form_submit_button("Predict")

if submit:
    # Ensure that income is not less than the requested loan amount
    if income < loan_amount:
        st.error("Annual income must be at least the loan amount.")
    else:
        # Build a DataFrame with the user inputs
        input_df = pd.DataFrame([{
            "person_age": age,
            "person_gender": gender,
            "person_education": education,
            "person_income": income,
            "person_emp_exp": emp_exp,
            "person_home_ownership": home,
            "loan_amnt": loan_amount,
            "loan_intent": loan_intent,
            "loan_int_rate": interest_rate,
            "loan_percent_income": percent_income,
            "cb_person_cred_hist_length": credit_history,
            "credit_score": credit_score,
            "previous_loan_defaults_on_file": previous_defaults
        }])

        # Use the pipeline to make a prediction
        prediction = pipeline.predict(input_df)[0]
        loan_status = "Approved" if prediction == 1 else "Rejected"
        st.success(f"Loan status predicted as: {loan_status}")

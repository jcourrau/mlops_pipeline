import streamlit as st
import pandas as pd
import joblib
import os

root = os.path.dirname(os.path.dirname(__file__))

# Carga modelo y scaler
model = joblib.load(f"{root}/models/loan_model.pkl")
scaler = joblib.load(f"{root}/models/scaler.pkl")

st.title("Predictor de Loan Approval")

with st.form("loan_form"):
    age = st.number_input("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
    income = st.number_input("Annual Income", 0.0, 1e6, 50000.0, step=1000.0)
    emp_exp = st.number_input("Employment Years", 0, 50, 5)
    home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_amnt = st.number_input("Loan Amount", 0.0, 1e6, 10000.0, step=500.0)
    loan_intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL",
                                                "VENTURE", "HOME_IMPROVEMENT", "DEBT_CONSOLIDATION"])
    loan_int_rate = st.number_input("Interest Rate (%)", 0.0, 100.0, 10.0)
    loan_pct_inc = st.number_input("Loan % of Income", 0.0, 200.0, 20.0)
    cred_hist = st.number_input("Credit History (yrs)", 0.0, 50.0, 5.0)
    credit_score = st.number_input("Credit Score", 300, 850, 650)
    prev_def = st.selectbox("Previous Defaults?", ["No", "Yes"])
    submitted = st.form_submit_button("Predict")

if submitted:
    # Validaciones simples
    errs = []
    if income < loan_amnt:
        errs.append("Income debe ≥ loan amount.")
    if loan_int_rate > 50:
        errs.append("Interest rate muy alto.")
    if errs:
        for e in errs:
            st.error(e)
    else:
        # Crea DataFrame
        df = pd.DataFrame({
            "person_age": [age],
            "person_gender": [gender],
            "person_education": [education],
            "person_income": [income],
            "person_emp_exp": [emp_exp],
            "person_home_ownership": [home],
            "loan_amnt": [loan_amnt],
            "loan_intent": [loan_intent],
            "loan_int_rate": [loan_int_rate],
            "loan_percent_income": [loan_pct_inc],
            "cb_person_cred_hist_length": [cred_hist],
            "credit_score": [credit_score],
            "previous_loan_defaults_on_file": [prev_def]
        })
        # Aplica mismo preprocessing de categorías y escalado (podrías extraer función)
        df_enc = pd.get_dummies(df, drop_first=True).reindex(columns=scaler.feature_names_in_, fill_value=0)
        X_scaled = scaler.transform(df_enc)
        pred = model.predict(X_scaled)[0]
        st.success(f"Loan status: **{pred}**")

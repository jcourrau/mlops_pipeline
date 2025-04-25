import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Loan Approval",
    page_icon="assets/piggy-bank-32.png",
    layout="wide"
)

# Load the trained prediction pipeline
with st.spinner("Loading model, please wait…"):
    pipeline = joblib.load("models/loan_pipeline.pkl")

# Initialize prediction history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Define table columns for history
history_columns = [
    "Age", "Income", "Loan Amount", "Interest Rate", "Credit Score", "Result"
]

# Create tabs for navigation
tabs = st.tabs(["Predict", "About"])

# Predict tab
with tabs[0]:
    st.header("Loan Approval Predictor")

    # Input form placed in the sidebar
    with st.sidebar.form("loan_form", clear_on_submit=True):
        st.subheader("Enter loan details:")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox(
            "Gender", ["male", "female"], format_func=lambda x: x.title()
        )
        education = st.selectbox(
            "Education Level", ["High School", "Bachelor", "Master", "PhD"]
        )
        income = st.number_input(
            "Annual Income", min_value=0.0, value=50000.0, step=1000.0
        )
        emp_exp = st.number_input(
            "Years of Employment", min_value=0, max_value=50, value=5
        )
        home = st.selectbox(
            "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"],
            format_func=lambda x: x.title()
        )
        loan_amount = st.number_input(
            "Loan Amount", min_value=0.0, value=10000.0, step=500.0
        )
        loan_intent = st.selectbox(
            "Loan Purpose", [
                "PERSONAL", "EDUCATION", "MEDICAL",
                "VENTURE", "HOME_IMPROVEMENT", "DEBT_CONSOLIDATION"
            ], format_func=lambda x: x.replace("_", " ").title()
        )
        interest_rate = st.number_input(
            "Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0
        )
        percent_income = st.number_input(
            "Loan as % of Income", min_value=0.0, max_value=200.0, value=20.0
        )
        credit_history = st.number_input(
            "Credit History (years)", min_value=0.0, max_value=50.0, value=5.0
        )
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=850, value=650
        )
        previous_defaults = st.selectbox("Previous Defaults?", ["No", "Yes"])
        submit = st.form_submit_button("Predict")

    if submit:
        # Basic validation: income must cover the loan amount
        if income < loan_amount:
            st.error("Annual income must be at least the loan amount.")
        else:
            # Build input DataFrame for prediction
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

            # Generate prediction
            prediction = pipeline.predict(input_df)[0]
            status = "Approved" if prediction == 1 else "Rejected"

            # Append summary to history
            st.session_state.history.append({
                "Age": age,
                "Income": income,
                "Loan Amount": loan_amount,
                "Interest Rate": interest_rate,
                "Credit Score": credit_score,
                "Result": status
            })

            # Display result prominently
            if status == "Approved":
                st.success("Load Approved ✅")
            else:
                st.error("Load Rejected ❌")

    # Display prediction history table
    st.subheader("Prediction History")
    history_df = pd.DataFrame(
        st.session_state.history,
        columns=history_columns
    )
    st.dataframe(history_df)

# About tab
with tabs[1]:
    st.header("About")
    st.markdown(
        "Loan Approval Predictor is a tool created by Jason Courrau. "
        "It simulates loan approval or rejection based on a trained model. "
        "Version 1.0."
    )
    st.markdown("---")
    st.markdown(
        "**Instructions:**\n"
        "1. Go to the **Predict** tab.\n"
        "2. Fill in your loan details in the sidebar.\n"
        "3. Click **Predict** to see the result.\n"
    )

import datetime
import platform
import subprocess

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

def get_debug_info():
    """
    Gather system and environment debug information.
    
    Collects various system metrics and information including:
    - Git commit hash
    - Build timestamp
    - System hostname
    - Platform details
    - Python version
    - Current server time
    - Disk usage
    - Available RAM
    - Running Docker containers
    
    Returns:
        pandas.DataFrame: Debug information with columns ['Metric', 'Value']
    """

    debug_data = {
        "Git Commit SHA": os.getenv("GIT_COMMIT", "Unknown"),
        "Build Time (UTC)": os.getenv("BUILD_TIME", "Unknown"),
        "Hostname": platform.node(),
        "Platform": platform.platform(),
        "Python Version": platform.python_version(),
        "Server Time (Local)": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Disk Usage": subprocess.getoutput("df -h / | tail -1 | awk '{print $5}'"),
        "RAM Free (MB)": subprocess.getoutput("free -m | awk '/Mem:/ {print $7}'"),
        "Running Containers": subprocess.getoutput("docker ps --format '{{.Names}}'"),
    }

    df = pd.DataFrame(debug_data.items(), columns=["Metric", "Value"])
    return df

# Page configuration
st.set_page_config(
    page_title="Loan Predictor",
    page_icon="assets/icon-32-white.png",
    layout="wide"
)

st.markdown("""
    <style>
    button[role="tablist"] {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained prediction pipeline
with st.spinner("Loading model, please wait…"):
    @st.cache_resource
    def load_pipeline():
        """
        Load and cache the trained machine learning pipeline from disk.
        The pipeline includes both preprocessing steps and the classifier.
        
        Returns:
            sklearn.pipeline.Pipeline: Loaded pipeline object containing preprocessor and classifier
        """
        return joblib.load("models/loan_pipeline.pkl")

    pipeline = load_pipeline()

# Initialize prediction history in the session state
if "history" not in st.session_state:
    st.session_state.history = []

# Define table columns for history
history_columns = [
    "Client ID", 
    "Age", 
    "Income",
    "Loan Amount", 
    "Interest Rate", 
    "Credit Score",
    "Result"
]

# Create tabs for navigation
tabs = st.tabs(["Predict", "Dashboard", "About"])

# Predict tab
with tabs[0]:
    st.header("Loan Approval Predictor")

    with st.form("loan_form", clear_on_submit=True):
        st.subheader("Enter loan details:")
        # Create two columns
        col1, col2, col3 = st.columns(3)

        with col1:
            client_id = st.text_input(
                "Client ID (8 characters)",
                max_chars=8,
                value="20703025"
            )
            education = st.selectbox(
                "Education Level", ["High School", "Bachelor", "Master", "PhD"]
            )
            home = st.selectbox(
                "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"],
                format_func=lambda x: x.title()
            )
            interest_rate = st.number_input(
                "Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0
            )
            credit_score = st.number_input(
                "Credit Score", min_value=300, max_value=850, value=650
            )

        with col2:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            emp_exp = st.number_input(
                "Years of Employment", min_value=0, max_value=50, value=5
            )
            loan_intent = st.selectbox(
                "Loan Purpose", [
                    "PERSONAL", "EDUCATION", "MEDICAL",
                    "VENTURE", "HOME_IMPROVEMENT", "DEBT_CONSOLIDATION"
                ], format_func=lambda x: x.replace("_", " ").title()
            )
            percent_income = st.number_input(
                "Loan as % of Income", min_value=0.0, max_value=200.0, value=20.0
            )
            credit_history = st.number_input(
                "Credit History (years)", min_value=0.0, max_value=50.0, value=5.0
            )

            # Predict Button
            submit = st.form_submit_button("Predict", use_container_width=True)

        with col3:
            gender = st.selectbox(
                "Gender", ["male", "female"], format_func=lambda x: x.title()
            )
            income = st.number_input(
                "Annual Income", min_value=0.0, value=50000.0, step=1000.0
            )
            loan_amount = st.number_input(
                "Loan Amount", min_value=0.0, value=10000.0, step=500.0
            )
            previous_defaults = st.selectbox("Previous Defaults?", ["No", "Yes"])

    # Prediction result and history table on the right column
    if submit:
        # Validate client ID
        if not client_id.isdigit():
            st.error("Client ID must contain only numbers.")
        elif len(client_id) != 8:
            st.error("Client ID must be exactly 8 digits long.")

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
                "Client ID": client_id,
                "Age": age,
                "Income": income,
                "Loan Amount": loan_amount,
                "Interest Rate": interest_rate,
                "Credit Score": credit_score,
                "Result": status
            })

            # Display the result
            if status == "Approved":
                st.success("Loan Approved ✅")
            else:
                st.error("Loan Rejected ❌")

    history_df = pd.DataFrame(
        st.session_state.history,
        columns=history_columns
    )

    # Display prediction history table
    st.subheader("Prediction History")
    st.dataframe(history_df)

# Dashboard tab
with tabs[1]:
    st.header("Dashboards")

    history_df = pd.DataFrame(
        st.session_state.history,
        columns=history_columns
    )

    if not history_df.empty:

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Approval Rate")
            fig_pie = px.pie(
                history_df,
                names="Result",
                hole=0.4,
                template="plotly_white",
                color_discrete_map={
                    "Approved": "#2ECC71",
                    "Rejected": "#E74C3C"
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Loan Distribution")
            st.text("Income vs Loan Amount")
            fig = px.scatter(
                history_df,
                x="Income",
                y="Loan Amount",
                color="Result",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Row 2: Full-width trend line (placeholder, update later)
        st.subheader("Approved vs Rejected Loans")
        st.text("by Credit Score Range")
        history_df["Credit Bin"] = pd.cut(
            history_df["Credit Score"],
            bins=[300, 500, 650, 750, 850]
        )
        history_df["Credit Bin"] = history_df["Credit Bin"].astype(str)
        fig_stack = px.histogram(
            history_df,
            x="Credit Bin",
            color="Result",
            barmode="stack",
            template="plotly_white"
        )
        st.plotly_chart(fig_stack, use_container_width=True)

        # Final row: full-width table
        st.subheader("Prediction History")
        st.dataframe(history_df)
    else:
        st.info("Generate some predictions first to see the dashboards.")

# About tab
with tabs[2]:
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

    st.title("🔧 Debug Info")

    debug_df = get_debug_info()
    st.table(debug_df)

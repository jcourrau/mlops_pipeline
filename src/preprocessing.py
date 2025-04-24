import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load Data
def load_data(path="data/loan_data.csv"):
    df = pd.read_csv(path)
    print(f"{df.shape[0]} rows loaded from:\n{path}")
    print(f"{df.head()}")
    return df

# Pre-process data as needed.
def preprocess(df):
    df = df.dropna()
    cat_cols = [
        "person_gender", "person_education", "person_home_ownership",
        "loan_intent", "previous_loan_defaults_on_file"
    ]
    df_cat = pd.get_dummies(df[cat_cols], drop_first=True)

    num_cols = [
        "person_age", "person_income", "person_emp_exp",
        "loan_amnt", "loan_int_rate", "loan_percent_income",
        "cb_person_cred_hist_length", "credit_score"
    ]
    scaler = StandardScaler()
    df_num = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

    X = pd.concat([df_num, df_cat], axis=1)
    y = df["loan_status"]

    print(f"Data preprocessed.\n")

    return X, y, scaler


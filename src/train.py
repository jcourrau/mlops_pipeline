import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

root = os.path.dirname(os.path.dirname(__file__))

# Load the data and remove any missing values
print("Reading data...")
df = pd.read_csv(f"{root}/data/loan_data.csv").dropna()

# Specify which columns are numeric and which are categorical
numeric_features = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score"
]
categorical_features = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file"
]

# Create a transformer that scales numerical data and encodes categorical data
preprocessor = ColumnTransformer([
    ("scale", StandardScaler(), numeric_features),
    ("encode", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
])

# Combine the preprocessor with a random forest classifier into a single pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])

# Split the dataset into training and testing subsets
X = df[numeric_features + categorical_features]
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the pipeline and save it to disk
print("Training the pipeline...")
pipeline.fit(X_train, y_train)
os.makedirs(f"{root}/models", exist_ok=True)
joblib.dump(pipeline, f"{root}/models/loan_pipeline.pkl")

print("Pipeline has been saved to models/loan_pipeline.pkl")

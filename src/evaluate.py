import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# load the pipeline that includes preprocessing and the classifier
root = os.path.dirname(os.path.dirname(__file__))
pipeline = joblib.load(f"{root}/models/loan_pipeline.pkl")

# load the raw dataset and drop any missing values
df = pd.read_csv(f"{root}/data/loan_data.csv").dropna()

# define which columns were used as inputs to the pipeline
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

# split the data into features and target
X = df[numeric_features + categorical_features]
y = df["loan_status"]

# split into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# use the pipeline to make predictions on the test set
y_pred = pipeline.predict(X_test)

# calculate accuracy and error rate
acc = accuracy_score(y_test, y_pred)
err_rate = 1 - acc

# ensure the reports directory exists
os.makedirs(f"{root}/reports", exist_ok=True)

# save metrics to a text file
metrics_path = f"{root}/reports/metrics.txt"
with open(metrics_path, "w") as f:
    f.write(f"Accuracy: {acc*100:.4f}\nError rate: {err_rate*100:.4f}\n")

# extract feature names from the preprocessor and classifier
preprocessor = pipeline.named_steps["preprocessor"]
feature_names = preprocessor.get_feature_names_out()
importance = pipeline.named_steps["classifier"].feature_importances_

# plot and save feature importance
plt.figure(figsize=(8, 6))
plt.barh(feature_names, importance)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(f"{root}/reports/feature_importance.png")

print(f"Report saved to:\n{root}/reports")
print(f"Accuracy: {acc*100:.2f}%\nError rate: {err_rate*100:.2f}%\n")

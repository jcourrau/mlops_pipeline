import joblib
import os
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from src.preprocessing import load_data, preprocess
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(__file__))
    model = joblib.load(f"{root}/models/loan_model.pkl")
    scaler = joblib.load(f"{root}/models/scaler.pkl")

    df = load_data(f"{root}/data/loan_data.csv")
    X, y, _ = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    err_rate = 1 - acc

    os.makedirs("reports", exist_ok=True)
    with open(f"{root}/reports/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nError rate: {err_rate:.4f}\n")

    imp = model.feature_importances_
    feat = X.columns
    plt.figure(figsize=(8,6))
    plt.barh(feat, imp)
    plt.tight_layout()
    plt.savefig(f"{root}/reports/feature_importance.png")

    print(f"Report saved on:\n{root}/reports")
    print(f"Accuracy: {acc*100:.2f}%\nError rate: {err_rate*100:.2f}%\n")

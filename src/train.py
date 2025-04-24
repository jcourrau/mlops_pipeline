import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.preprocessing import load_data, preprocess

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(__file__))
    data_path = f"{root}/data/loan_data.csv"
    df = load_data(data_path)
    X, y, scaler = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"{root}/models/loan_model.pkl")
    joblib.dump(scaler, f"{root}/models/scaler.pkl")

    print(f"Model & scaler serialized in:\n {root}/models")

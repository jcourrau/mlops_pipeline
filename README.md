# Loan Approval Predictor - MLOps Final Project

## Project Overview

This project consists of a full Machine Learning Operations (MLOps) pipeline to predict loan approvals based on customer information. The solution integrates machine learning modeling, automated re-training, data and model versioning, Docker-based deployment, and a web interface built with Streamlit.

---

## Author / Student

- Jason Courrau

---

## Project Structure

```
.
├── app.py                   # Streamlit web app
├── src/
│   ├── download_data.py      # Download data from S3 bucket
│   ├── train.py              # Train ML model
│   ├── evaluate.py           # Evaluate ML model
├── Dockerfile                # Docker container setup
├── .github/workflows/main.yml # GitHub Actions for CI/CD
├── .deepsource.toml          # DeepSource static code analysis config
├── mlops.cent-0.com.conf     # Nginx configuration for HTTPS proxy
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation (this file)
└── data/, models/, reports/  # Folders created during runtime
```

---

## Main Features

- **ML Model**: Random Forest Classifier for loan approval prediction.
- **Interface**: Streamlit app with 3 tabs (Predict, Dashboard, About).
- **Containerization**: Dockerfile to create production-ready containers.
- **CI/CD**: GitHub Actions automate retraining, DVC versioning, DockerHub publishing, and EC2 deployment.
- **Data/Model Versioning**: DVC integrated with AWS S3.
- **Static Code Analysis**: DeepSource configuration for Python linting.
- **Nginx Proxy**: Configuration to support HTTPS and WebSocket proxying for the app.

---

## Usage Instructions

These instructions allow you to replicate the code and run the project on your own environment. If you simply want to use the application, you can access it directly at [https://mlops.cent-o.com/](https://mlops.cent-o.com/).

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mlops-loan-predictor.git
cd mlops-loan-predictor
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Download dataset

```bash
python src/download_data.py
```

### 4. Train and evaluate model

```bash
python src/train.py
python src/evaluate.py
```

### 5. Run Streamlit app locally

```bash
streamlit run app.py
```

### 6. Build and Run with Docker (Production Setup)

```bash
docker build -t loan-approval-app .
docker run -p 8501:8501 loan-approval-app
```

---

## Inputs and Outputs

| Input Field           | Description                         |
| --------------------- | ----------------------------------- |
| Client ID             | 8-digit unique ID                   |
| Age                   | Person's age                        |
| Income                | Annual income                       |
| Loan Amount           | Requested loan amount               |
| Interest Rate         | Interest rate (%)                   |
| Credit Score          | FICO score                          |
| Gender                | Male/Female                         |
| Education             | Highest education level             |
| Home Ownership        | Rent/Own/Mortgage/Other             |
| Loan Purpose          | Reason for loan                     |
| Years of Employment   | Experience in years                 |
| Loan as % of Income   | Loan amount as percentage of income |
| Credit History Length | Length of credit history in years   |
| Previous Defaults     | Whether the person defaulted before |

**Output**: Approved or Rejected.

---

## Branches Used

- **develop**: Main branch for active development.
- **staging**: Branch for staging new tested features.
- **main**: Branch for production releases.

---

## Public Links

- **GitHub Repository**: https\://github.com/jcourrau/mlops\_pipeline/
- **App URL**: [https://mlops.cent-o.com/](https://mlops.cent-o.com/)

---

## Technologies Used

- **Python 3.10**
- **Streamlit**
- **scikit-learn**
- **DVC**
- **AWS S3**
- **Docker**
- **GitHub Actions**
- **Nginx**
- **DeepSource**

---

## Notes

- Data and model files are automatically versioned and stored in AWS S3.
- GitHub Actions pipeline handles automatic model retraining and deployment after each `develop` branch push.
- Inputs are validated both in the Streamlit interface and backend.
- Static code analysis is set up via DeepSource.

---

## Contact

If you encounter issues or have questions, feel free to reach out to Jason Courrau.


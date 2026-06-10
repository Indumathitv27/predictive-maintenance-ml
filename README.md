---
title: Predictive Maintenance ML
emoji: ⚙️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ⚙️ Predictive Maintenance AI System

> **End-to-end ML pipeline predicting turbofan engine Remaining Useful Life (RUL) using LightGBM, Optuna, MLflow, FastAPI, Streamlit, and Snowflake.**

---

## 🔧 What it does

Predicts how many cycles remain before a turbofan engine fails — enabling proactive maintenance scheduling instead of reactive repairs.

1. Ingests NASA CMAPS sensor data (20,631 training cycles)
2. Engineers 48 time-series features — rolling mean, rolling std, lag features
3. Trains RandomForest, XGBoost, and LightGBM with 5-fold cross-validation
4. Tunes LightGBM with Optuna Bayesian optimization (50 trials)
5. Tracks all experiments with MLflow
6. Serves predictions via FastAPI REST endpoint
7. Logs every prediction to Snowflake in real time
8. Displays fleet health on a live Streamlit dashboard

---

## 📊 Model Performance

| Model | Val RMSE | Val R2 | Status |
|---|---|---|---|
| RandomForest | 11.34 | 0.932 | Baseline |
| XGBoost | 10.85 | 0.9377 | Baseline |
| LightGBM | 10.71 | 0.9393 | Baseline |
| LightGBM (Optuna Tuned) | 10.59 | 0.9407 | Best |

---

## 🏗️ Architecture
NASA CMAPS Dataset
↓
Feature Engineering (48 features)
↓
Model Training — RF, XGBoost, LightGBM
↓
Optuna Hyperparameter Tuning (50 trials)
↓
MLflow Experiment Tracking
↓
FastAPI REST Endpoint (/predict, /predict/batch)
↓
Snowflake — real-time prediction logging
↓
Streamlit Dashboard — fleet health monitoring

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data & Features | Python, Pandas, NumPy, scikit-learn |
| ML Models | LightGBM, XGBoost, RandomForest |
| Hyperparameter Tuning | Optuna (Bayesian optimization) |
| Experiment Tracking | MLflow |
| Backend API | FastAPI, Uvicorn |
| Data Warehouse | Snowflake |
| Frontend | Streamlit |
| Containerization | Docker |
| Deployment | Hugging Face Spaces |
| Cloud Mapping | GCP Vertex AI, AWS SageMaker |

---

## 🚀 Setup & Installation

```bash
git clone https://github.com/Indumathitv27/predictive-maintenance-ml.git
cd predictive-maintenance-ml

conda create -n pred-maintenance python=3.10
conda activate pred-maintenance
pip install -r requirements.txt
```

Create `.env`:
GROQ_API_KEY=your_key
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_DATABASE=PREDICTIVE_MAINTENANCE
SNOWFLAKE_SCHEMA=ML_PIPELINE
SNOWFLAKE_WAREHOUSE=COMPUTE_WH

Run:
```bash
# Terminal 1
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2
streamlit run dashboard/app.py
```

---

## 🗂️ Project Structure

| Folder | Purpose |
|---|---|
| `features/` | Data ingestion, preprocessing, feature engineering |
| `models/` | Model training, Optuna tuning, MLflow tracking |
| `api/` | FastAPI backend, Snowflake logger |
| `dashboard/` | Streamlit frontend |
| `data/raw/` | NASA CMAPS dataset (gitignored) |
| `data/processed/` | Engineered features (gitignored) |
| `mlruns/` | MLflow experiment artifacts |

---

## ☁️ Production Mapping

| Local | AWS | GCP |
|---|---|---|
| LightGBM model | SageMaker endpoint | Vertex AI endpoint |
| FastAPI | API Gateway + Lambda | Cloud Run |
| MLflow | SageMaker Experiments | Vertex AI Experiments |
| Snowflake | Snowflake on AWS | Snowflake on GCP |
| Docker | ECR + ECS Fargate | Artifact Registry + GKE |

---

## ⚠️ Disclaimer

For research and educational purposes only.

---

## 👩‍💻 Author

**Indumathi Tamil Selvi Varadharajan**
M.S. Data Science — University at Buffalo
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/indumathitv2702/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Indumathitv27/)
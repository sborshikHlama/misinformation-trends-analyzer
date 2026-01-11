# Fake News Style Detector

**Detecting Misinformation Patterns Through Topic Modeling and Style Analysis**

An end-to-end Machine Learning application that analyzes news articles using BERTopic for topic discovery and Logistic Regression for fake/real classification. The system identifies 572 thematic categories and provides probability scores for prioritizing fact-checking efforts.


## Business Goal

A solution designed for **fact-checkers, journalists, and media analysts** to quickly screen incoming news articles before manual review.

---

## Project Structure

```
misinformation-trends-analyzer/
├── src/
│   ├── backend/                  # FastAPI REST API
│   │   ├── main.py               # API endpoints and lifespan
│   │   ├── models.py             # Pydantic schemas
│   │   └── services.py           # ML model service
│   └── frontend/
│       └── app.py                # Streamlit UI
├── bert_fake_true/               # BERTopic model artifacts
├── log_reg_fake_true/            # TF-IDF + LogReg model
├── topic_mapping.json            # Topic ID to names (572 topics)
├── data/
│   ├── Fake.csv
│   └── True.csv
├── docs/
│   ├── images/                   # MLflow screenshots
│   ├── business_report.md
│   ├── data_processing.md
│   ├── model_development.md
│   ├── failed_approaches.md
│   └── failure_analysis.md
├── tests/
│   └── test_app.py               # 20 unit tests
├── k8s/
│   ├── backend.yaml
│   └── frontend.yaml
├── mlruns/                       # MLflow experiments
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Docker Desktop
- KIND and kubectl (for Kubernetes deployment)

### Option A: Docker Compose

```bash
docker compose up --build
```

### Option B: Kubernetes (KIND)

```bash
kind create cluster --name fake-news

docker build -f Dockerfile.backend -t fake-news-backend:latest .
docker build -f Dockerfile.frontend -t fake-news-frontend:latest .
kind load docker-image fake-news-backend:latest --name fake-news
kind load docker-image fake-news-frontend:latest --name fake-news

kubectl apply -f k8s/
kubectl port-forward svc/frontend 8501:80
```

### Access

- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000/docs

---

## Model Overview

### Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle - Fake and Real News Dataset |
| Total Articles | 44,898 |
| Fake News | 23,481 |
| Real News | 21,417 |

### Results

| Metric | Value |
|--------|-------|
| Classification Accuracy | 98% |
| F1 Score | 0.98 |
| ROC-AUC | 0.99 |
| Topics Discovered | 572 |
| Outlier Rate | 12% |
| Inference Time | <1s |

---

## Example Inputs and Outputs

| Input | Output | Topic |
|-------|--------|-------|
| Reuters reports Federal Reserve raised rates... | REAL (97%) | Economic Trends |
| BREAKING: Hillary Clinton EXPOSED... | FAKE (97%) | Clinton Scandals |

---

---

## Documentation

| Document | Description |
|----------|-------------|
| [Business Report](docs/business_report.md) | ML problem framing, success metrics |
| [Data Processing](docs/data_processing.md) | Preprocessing decisions |
| [Model Development](docs/model_development.md) | MLflow experiments |
| [Failed Approaches](docs/failed_approaches.md) | 5 failed experiments |
| [Failure Analysis](docs/failure_analysis.md) | Limitations, edge cases |

---




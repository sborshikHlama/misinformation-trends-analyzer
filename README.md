# Misinformation Trends Analyzer

**Topic discovery and writing-style screening for news articles**, shipped as a containerised ML service.

The system pairs unsupervised topic modelling (BERTopic) with a supervised fake/real style classifier
(TF-IDF + logistic regression) behind a FastAPI backend and a Streamlit UI. It is meant to help fact-checkers
and media analysts decide which incoming articles to read first, not to decide what is true.

> The classifier learns **writing style**, not factual accuracy. See [Limitations](#limitations).

## Architecture

```
                 ┌───────────────┐   POST /analyze    ┌──────────────────────────────┐
 Browser ──────► │ Streamlit UI  │ ─────────────────► │ FastAPI backend              │
                 │   :8501       │ ◄───────────────── │   :8000                      │
                 └───────────────┘   topic + verdict  │  ├─ BERTopic (topic, keywords)│
                                                      │  └─ TF-IDF + LogReg (fake/real)│
                                                      └──────────────┬───────────────┘
                                                                     │ model artifacts
                                                       bert_fake_true/  log_reg_fake_true/
```

| Layer | Details |
|-------|---------|
| Topic model | BERTopic: `all-MiniLM-L6-v2` embeddings → UMAP → HDBSCAN → c-TF-IDF, 572 named topics in the shipped model |
| Classifier | TF-IDF (5,000 features) + logistic regression, returns class probabilities |
| API | FastAPI, pydantic schemas, `/health`, `/analyze`, `/topics`, OpenAPI docs at `/docs` |
| UI | Streamlit, talks to the API through `API_URL` |
| Packaging | One Dockerfile per service (non-root user, health checks), `docker-compose.yml`, Kubernetes manifests |
| Experiment tracking | MLflow, with logged runs for the good and the failed configurations |
| CI/CD | Lint, format check, unit tests, image build; GitLab CI and GitHub Actions definitions |

## Quick start

### Docker Compose

```bash
docker compose up --build
```

- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

### Kubernetes (KIND)

```bash
kind create cluster --name fake-news

docker build -f Dockerfile.backend  -t fake-news-backend:latest  .
docker build -f Dockerfile.frontend -t fake-news-frontend:latest .
kind load docker-image fake-news-backend:latest  --name fake-news
kind load docker-image fake-news-frontend:latest --name fake-news

kubectl apply -f k8s/
kubectl port-forward svc/frontend 8501:80
```

### Without Docker

```bash
pip install -r requirements.txt

# backend (from the repo root)
cd src/backend
BERTOPIC_MODEL_PATH=../../bert_fake_true LOGREG_MODEL_PATH=../../log_reg_fake_true uvicorn main:app --port 8000

# frontend, in a second terminal
API_URL=http://localhost:8000 streamlit run src/frontend/app.py
```

### Example request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Reuters reports the Federal Reserve raised interest rates by a quarter point on Wednesday..."}'
```

```json
{
  "text_preview": "Reuters reports the Federal Reserve raised interest rates ...",
  "topic": { "topic_id": 12, "topic_name": "Economic Trends", "confidence": 0.8, "keywords": ["fed", "rates", "..."] },
  "classification": { "prediction": "REAL", "fake_probability": 0.03, "real_probability": 0.97, "confidence": 0.97 }
}
```

(Values are illustrative; the shape matches `AnalyzeResponse` in `src/backend/models.py`.)

## Model overview

**Data:** Kaggle *Fake and Real News* dataset, 44,898 articles (23,481 fake, 21,417 real), English, 2016–2017.

| Metric (hold-out split of that dataset) | Value |
|-----------------------------------------|-------|
| Accuracy | 0.98 |
| F1 | 0.98 |
| ROC-AUC | 0.99 |
| Topic outlier rate | 12% |
| Inference time | < 1 s per article |

Experiments, including the configurations that failed (HDBSCAN cluster sizes, TF-IDF vocabulary size, regularisation),
are written up with MLflow screenshots in [`docs/model_development.md`](docs/model_development.md) and
[`docs/failed_approaches.md`](docs/failed_approaches.md).

## Engineering

- **Tests:** 20 unit tests for the service layer and the API (`tests/test_app.py`). They mock the model artifacts, so they run in seconds without GPU or model files.
- **CI:** `flake8` and `black --check` on the application code, then `pytest`, then a Docker build of both images. Defined for GitLab CI (`.gitlab-ci.yml`, which also pushes images to the registry and has a manual Kubernetes deploy job) and for GitHub Actions (`.github/workflows/ci.yml`).
- **Containers:** slim Python base, non-root runtime user, health checks on both services, read-only model mounts in Compose.
- **Configuration:** model locations and the API URL come from environment variables.
- **Tracking:** `scripts/test_models_mlflow.py` retrains the candidate configurations and logs parameters and metrics to MLflow (`mlflow ui --port 5001`).

## Limitations

The headline numbers come from a hold-out split of the same dataset used for training, so they describe in-distribution
performance only. They show that the model separates that dataset's two kinds of articles, not that it can judge news in general.

- **Style is not truth.** A well-written falsehood reads as REAL; blunt or emotional true reporting can read as FAKE.
- **Dataset artefacts.** The real articles are dominated by one wire-service style, which the classifier picks up. Inside the service the label mapping is flipped to match that learned behaviour (see the comment in `src/backend/services.py`).
- **Temporal bias.** Training data is from 2016–2017; later events and vocabulary are unknown to the topic model.
- **English only**; very long inputs are slow.
- **No fact-checking or source verification.**

Failure cases with concrete inputs are in [`docs/failure_analysis.md`](docs/failure_analysis.md).

## Documentation

| Document | Content |
|----------|---------|
| [Business report](docs/business_report.md) | Problem framing, success metrics, stakeholders |
| [Data processing](docs/data_processing.md) | Preprocessing decisions |
| [Model development](docs/model_development.md) | MLflow experiments and hyperparameter choices |
| [Failed approaches](docs/failed_approaches.md) | Experiments that did not work and why |
| [Failure analysis](docs/failure_analysis.md) | Edge cases and limitations |

## Repository layout

```
├── src/backend/            # FastAPI service (main.py, models.py, services.py)
├── src/frontend/app.py     # Streamlit UI
├── bert_fake_true/         # BERTopic artifacts
├── log_reg_fake_true/      # TF-IDF vectoriser + classifier
├── topic_mapping.json      # topic id → readable name
├── tests/                  # unit tests
├── scripts/                # MLflow experiment runner
├── k8s/                    # Kubernetes manifests
├── docs/                   # reports and MLflow screenshots
├── Fake_News_model_training.ipynb, models_training.ipynb
├── Dockerfile.backend, Dockerfile.frontend, docker-compose.yml
└── .gitlab-ci.yml, .github/workflows/ci.yml
```

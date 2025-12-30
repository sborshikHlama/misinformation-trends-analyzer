# 📰 Fake News Style Detector

A web application that uses BERTopic and Logistic Regression to analyze news text and detect if it's written in a fake news style.

## 🚀 Features

- **Topic Detection**: BERTopic identifies topic clusters
- **Fake/Real Classification**: Logistic Regression classifies news
- **REST API**: FastAPI backend
- **Web UI**: Streamlit frontend

## 🏗️ Project Structure

```
├── src/
│   ├── backend/            # Backend (FastAPI)
│   │   ├── main.py
│   │   ├── models.py
│   │   └── services.py
│   └── frontend/           # Frontend (Streamlit)
│       └── app.py
├── k8s/                    # Kubernetes configs
│   ├── backend.yaml
│   └── frontend.yaml
├── tests/
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
└── requirements.txt
```

## 🛠️ Installation

### Local

```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1 - Backend
PYTHONPATH=src/backend BERTOPIC_MODEL_PATH=./bert_fake_true LOGREG_MODEL_PATH=./log_reg_fake_true uvicorn src.backend.main:app --reload --port 8000

# Terminal 2 - Frontend
streamlit run src/frontend/app.py
```

- Frontend: http://localhost:8501
- Backend API: http://localhost:8000/docs

### Docker Compose

```bash
docker-compose up --build
```

## ☸️ Kubernetes (Kind)

### 1. Create cluster
```bash
kind create cluster --name fake-news
```

### 2. Build and load images
```bash
docker build -f Dockerfile.backend -t fake-news-backend:latest .
docker build -f Dockerfile.frontend -t fake-news-frontend:latest .

kind load docker-image fake-news-backend:latest --name fake-news
kind load docker-image fake-news-frontend:latest --name fake-news
```

### 3. Deploy
```bash
kubectl apply -f k8s/backend.yaml
kubectl apply -f k8s/frontend.yaml
```

### 4. Access
```bash
kubectl port-forward svc/frontend 8501:80
```
Open http://localhost:8501

### Useful commands
```bash
kubectl get pods
kubectl get svc
kubectl logs -l app=backend
kubectl delete -f k8s/
kind delete cluster --name fake-news
```

## 📊 API Endpoints

- `GET /health` - Health check
- `POST /analyze` - Analyze text
- `GET /topics` - Get topic mapping
- `GET /docs` - Swagger docs

## 📝 License

MIT License

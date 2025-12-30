import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    TopicMappingResponse,
)
from services import model_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load models on startup
    bertopic_path = Path(
        os.getenv("BERTOPIC_MODEL_PATH", "./bert_fake_true")
    )
    logreg_path = Path(
        os.getenv("LOGREG_MODEL_PATH", "./log_reg_fake_true")
    )
    try:
        model_service.load_models(bertopic_path=bertopic_path, logreg_path=logreg_path)
    except Exception as e:
        logger.warning(f"Failed to load models: {e}")
    yield


app = FastAPI(
    title="Fake News Detector API",
    description="API for detecting fake news using BERTopic and Logistic Regression",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check the health status of the API and models."""
    return model_service.get_health_status()


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_text(request: AnalyzeRequest):

    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        return model_service.analyze_text(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/topics", response_model=TopicMappingResponse, tags=["Topics"])
async def get_topics():
    # get the current topic mapping
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return model_service.get_topic_mapping()


@app.get("/", tags=["Root"])
async def root():
    # rot endpoint
    return {"message": "Fake News Detector API", "docs": "/docs"}

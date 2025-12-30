from pydantic import BaseModel, Field
from typing import List, Optional


class AnalyzeRequest(BaseModel):
    # request model for text analysis
    text: str = Field(..., min_length=1, description="Text to analyze")


class TopicInfo(BaseModel):
    # topic information from BERTopic analysis
    topic_id: int
    topic_name: str
    confidence: float
    keywords: List[str]


class ClassificationInfo(BaseModel):
    # cassification result from Logistic Regression
    prediction: str  # "FAKE" or "REAL"
    fake_probability: float
    real_probability: float
    confidence: float


class AnalyzeResponse(BaseModel):
    # response model for text analysis
    text_preview: str
    topic: TopicInfo
    classification: ClassificationInfo


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    bertopic_loaded: bool
    logreg_loaded: bool
    topic_mapping_loaded: bool


class TopicMappingResponse(BaseModel):
    # response for topic mapping endpoint
    total_topics: int
    mapping: dict


class ErrorResponse(BaseModel):
    # Error response model
    detail: str
    error_type: Optional[str] = None

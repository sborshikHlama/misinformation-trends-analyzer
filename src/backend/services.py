import os
import json
import pickle
import logging
from typing import Dict, Any, Optional

import numpy as np
import scipy.sparse as sp
import torch
from bertopic import BERTopic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MONKEY PATCH for scipy.sparse.diags ==========
_original_diags = sp.diags


def patched_diags(diagonals, offsets=0, shape=None, format=None, dtype=None):
    """Patch for scipy.sparse.diags that fixes BERTopic loading issue."""
    if isinstance(diagonals, torch.Tensor):
        diagonals = diagonals.cpu().numpy().flatten()
    elif sp.issparse(diagonals):
        diagonals = diagonals.diagonal()
    diagonals = np.asarray(diagonals).flatten()
    if offsets is None or (
        isinstance(offsets, (list, tuple, np.ndarray)) and len(offsets) == 0
    ):
        offsets = 0
    return _original_diags(
        diagonals, offsets=offsets, shape=shape, format=format, dtype=dtype
    )


sp.diags = patched_diags
# ========== END MONKEY PATCH ==========


class ModelService:
    # service for loading and using ML models

    def __init__(self):
        self.bertopic_model: Optional[BERTopic] = None
        self.vectorizer = None
        self.classifier = None
        self.topic_mapping: Dict[int, str] = {}
        self._models_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._models_loaded

    def load_models(
        self,
        bertopic_path: str = "bert_fake_true",
        logreg_path: str = "log_reg_fake_true",
    ) -> None:
        # load all models from specified paths.
        logger.info(f"Loading BERTopic model from {bertopic_path}...")
        self.bertopic_model = BERTopic.load(bertopic_path)
        logger.info("BERTopic model loaded!")

        logger.info(f"Loading LogReg model from {logreg_path}...")
        with open(os.path.join(logreg_path, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(os.path.join(logreg_path, "classifier.pkl"), "rb") as f:
            self.classifier = pickle.load(f)
        logger.info("LogReg model loaded!")

        self._load_topic_mapping(bertopic_path)
        self._models_loaded = True

    def _load_topic_mapping(self, bertopic_path: str) -> None:
        # lad topic name mapping from JSON file
        mapping_files = [
            "topic_mapping.json",
            "/models/topic_mapping.json",
            os.path.join(bertopic_path, "topic_mapping.json"),
        ]
        for mapping_file in mapping_files:
            if os.path.exists(mapping_file):
                with open(mapping_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self.topic_mapping = {int(k): v for k, v in raw.items()}
                logger.info(f"Topic mapping loaded from: {mapping_file}")
                return
        self.topic_mapping = {}

    def analyze_text(self, text: str) -> Dict[str, Any]:
        # anayze text using both models
        if not self._models_loaded:
            raise RuntimeError("Models not loaded")

        # BERTopic
        topics, probs = self.bertopic_model.transform([text])
        topic_id = int(topics[0])
        if probs is not None:
            if hasattr(probs, "shape"):  # numpy array
                if len(probs.shape) > 1:
                    topic_prob = float(probs[0].max())
                else:
                    topic_prob = float(probs[0])
            else:  # assume list/tuple
                if isinstance(probs[0], (list, tuple)):
                    topic_prob = float(max(probs[0]))
                else:
                    topic_prob = float(probs[0])
        else:
            topic_prob = 1.0
        topic_name = self.topic_mapping.get(topic_id, f"Topic {topic_id}")
        keywords = []
        if topic_id != -1:
            try:
                topic = self.bertopic_model.get_topic(topic_id)
                keywords = [word for word, _ in topic[:10]]
            except Exception:
                pass

        # LogReg
        text_vec = self.vectorizer.transform([text])
        pred = int(self.classifier.predict(text_vec)[0])
        proba = self.classifier.predict_proba(text_vec)[0]

        result = {
            "text_preview": text[:500] + "..." if len(text) > 500 else text,
            "topic": {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "confidence": topic_prob,
                "keywords": keywords,
            },
            "classification": {
                "prediction": "FAKE" if pred == 1 else "REAL",
                "fake_probability": float(proba[1]),
                "real_probability": float(proba[0]),
                "confidence": float(max(proba)),
            },
        }

        return result

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._models_loaded else "unhealthy",
            "models_loaded": self._models_loaded,
            "bertopic_loaded": self.bertopic_model is not None,
            "logreg_loaded": self.vectorizer is not None,
            "topic_mapping_loaded": len(self.topic_mapping) > 0,
        }

    def get_topic_mapping(self) -> Dict[str, Any]:
        return {"total_topics": len(self.topic_mapping), "mapping": self.topic_mapping}


model_service = ModelService()

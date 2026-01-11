import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

import pytest
from unittest.mock import patch, MagicMock, mock_open
from fastapi.testclient import TestClient

from main import app
from services import ModelService
from models import AnalyzeRequest


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_model_service():
    with patch('main.model_service') as mock_service:
        yield mock_service


class TestModelService:
    def setup_method(self):
        self.service = ModelService()

    def test_initial_state(self):
        assert self.service.is_loaded is False
        assert self.service.bertopic_model is None
        assert self.service.vectorizer is None
        assert self.service.classifier is None

    @patch('services.BERTopic.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    @patch('services.Path.exists')
    def test_load_models_success(self, mock_exists, mock_pickle, mock_file, mock_bertopic):
        mock_exists.return_value = False  # No topic mapping file
        mock_bertopic.return_value = MagicMock()
        mock_pickle.side_effect = [MagicMock(), MagicMock()]  # vectorizer, classifier

        self.service.load_models("fake_bert_path", "fake_logreg_path")

        assert self.service.is_loaded is True
        assert self.service.bertopic_model is not None

    def test_get_health_status_not_loaded(self):
        health = self.service.get_health_status()
        assert health["status"] == "unhealthy"
        assert health["models_loaded"] is False

    def test_analyze_text_without_loading(self):
        with pytest.raises(RuntimeError):
            self.service.analyze_text("test text")

    @patch('services.BERTopic.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    @patch('services.Path.exists')
    def test_load_topic_mapping(self, mock_exists, mock_pickle, mock_file, mock_bertopic):
        mock_exists.side_effect = lambda: True  # First call returns True
        mock_file.return_value.read.return_value = '{"0": "Test Topic"}'
        mock_bertopic.return_value = MagicMock()
        mock_pickle.side_effect = [MagicMock(), MagicMock()]

        self.service.load_models()

        assert self.service.topic_mapping == {0: "Test Topic"}

    @patch('services.BERTopic.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    @patch('services.Path.exists')
    def test_analyze_text_success(self, mock_exists, mock_pickle, mock_file, mock_bertopic):
        # Mock models
        mock_bertopic_model = MagicMock()
        mock_bertopic_model.transform.return_value = ([0], [[0.8, 0.2]])
        mock_bertopic_model.get_topic.return_value = [("word1", 0.5), ("word2", 0.3)]
        mock_bertopic.return_value = mock_bertopic_model

        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = MagicMock()
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = [1]  # pred=1 means REAL in our model
        mock_classifier.predict_proba.return_value = [[0.3, 0.7]]  # proba[0]=fake, proba[1]=real

        mock_pickle.side_effect = [mock_vectorizer, mock_classifier]
        mock_exists.return_value = False

        self.service.load_models()
        result = self.service.analyze_text("This is a test text about politics.")

        assert result["text_preview"] == "This is a test text about politics."
        assert result["topic"]["topic_id"] == 0
        assert result["topic"]["topic_name"] == "Topic 0"
        assert result["topic"]["confidence"] == 0.8
        assert result["topic"]["keywords"] == ["word1", "word2"]
        assert result["classification"]["prediction"] == "REAL"  # pred=1 means REAL
        assert result["classification"]["fake_probability"] == 0.3  # proba[0]
        assert result["classification"]["real_probability"] == 0.7  # proba[1]
        assert result["classification"]["confidence"] == 0.7

    @patch('services.BERTopic.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    @patch('services.Path.exists')
    def test_analyze_text_long_preview(self, mock_exists, mock_pickle, mock_file, mock_bertopic):
        mock_bertopic_model = MagicMock()
        mock_bertopic_model.transform.return_value = ([0], [[0.8]])
        mock_bertopic.return_value = mock_bertopic_model
        mock_vectorizer = MagicMock()
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = [0]
        mock_classifier.predict_proba.return_value = [[0.6, 0.4]]
        mock_pickle.side_effect = [mock_vectorizer, mock_classifier]
        mock_exists.return_value = False

        self.service.load_models()
        long_text = "A" * 600
        result = self.service.analyze_text(long_text)

        assert result["text_preview"].endswith("...")
        assert len(result["text_preview"]) == 503  # 500 + "..."

    @patch('services.BERTopic.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    @patch('services.Path.exists')
    def test_get_topic_mapping(self, mock_exists, mock_pickle, mock_file, mock_bertopic):
        mock_exists.side_effect = lambda: True  # First call returns True
        mock_file.return_value.read.return_value = '{"1": "Politics", "2": "Sports"}'
        mock_bertopic.return_value = MagicMock()
        mock_pickle.side_effect = [MagicMock(), MagicMock()]

        self.service.load_models()
        mapping = self.service.get_topic_mapping()

        assert mapping["total_topics"] == 2
        assert mapping["mapping"] == {1: "Politics", 2: "Sports"}

    @patch('services.BERTopic.load')
    def test_load_models_bertopic_failure(self, mock_bertopic):
        mock_bertopic.side_effect = Exception("BERTopic load failed")

        with pytest.raises(Exception):
            self.service.load_models()

        assert self.service.is_loaded is False

    @patch('services.BERTopic.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    @patch('services.Path.exists')
    def test_load_models_pickle_failure(self, mock_exists, mock_pickle, mock_file, mock_bertopic):
        mock_bertopic.return_value = MagicMock()
        mock_pickle.side_effect = Exception("Pickle load failed")

        with pytest.raises(Exception):
            self.service.load_models()

        assert self.service.is_loaded is False


class TestAPIEndpoints:
    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data

    def test_health_endpoint_models_not_loaded(self, client, mock_model_service):
        mock_model_service.is_loaded = False
        mock_model_service.get_health_status.return_value = {
            "status": "unhealthy",
            "models_loaded": False,
            "bertopic_loaded": False,
            "logreg_loaded": False,
            "topic_mapping_loaded": False,
        }

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["models_loaded"] is False

    def test_health_endpoint_models_loaded(self, client, mock_model_service):
        mock_model_service.is_loaded = True
        mock_model_service.get_health_status.return_value = {
            "status": "healthy",
            "models_loaded": True,
            "bertopic_loaded": True,
            "logreg_loaded": True,
            "topic_mapping_loaded": True,
        }

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True

    def test_analyze_endpoint_success(self, client, mock_model_service):
        mock_model_service.is_loaded = True
        mock_result = {
            "text_preview": "Test text",
            "topic": {"topic_id": 1, "topic_name": "Politics", "confidence": 0.9, "keywords": ["politics", "election"]},
            "classification": {"prediction": "REAL", "fake_probability": 0.2, "real_probability": 0.8, "confidence": 0.8}
        }
        mock_model_service.analyze_text.return_value = mock_result

        response = client.post("/analyze", json={"text": "This is a test article about politics."})
        assert response.status_code == 200
        data = response.json()
        assert data == mock_result

    def test_analyze_endpoint_models_not_loaded(self, client, mock_model_service):
        mock_model_service.is_loaded = False

        response = client.post("/analyze", json={"text": "Test text"})
        assert response.status_code == 503
        data = response.json()
        assert "Models not loaded" in data["detail"]

    def test_analyze_endpoint_invalid_request(self, client):
        response = client.post("/analyze", json={"invalid": "data"})
        assert response.status_code == 422  # Validation error

    def test_analyze_endpoint_empty_text(self, client):
        response = client.post("/analyze", json={"text": ""})
        assert response.status_code == 422  # Validation error

    def test_topics_endpoint_success(self, client, mock_model_service):
        mock_model_service.is_loaded = True
        mock_mapping = {"total_topics": 2, "mapping": {"0": "Sports", "1": "Politics"}}
        mock_model_service.get_topic_mapping.return_value = mock_mapping

        response = client.get("/topics")
        assert response.status_code == 200
        data = response.json()
        assert data == mock_mapping

    def test_topics_endpoint_models_not_loaded(self, client, mock_model_service):
        mock_model_service.is_loaded = False

        response = client.get("/topics")
        assert response.status_code == 503
        data = response.json()
        assert "Models not loaded" in data["detail"]

    def test_analyze_endpoint_analysis_failure(self, client, mock_model_service):
        mock_model_service.is_loaded = True
        mock_model_service.analyze_text.side_effect = Exception("Model inference failed")

        response = client.post("/analyze", json={"text": "Test text"})
        assert response.status_code == 500
        data = response.json()
        assert "Model inference failed" in data["detail"]
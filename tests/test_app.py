import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch, MagicMock
from app import load_model

class TestLoadModel:
    def setup_method(self):
        # Clear Streamlit cache before each test to avoid caching between tests
        import streamlit as st
        st.cache_resource.clear()

    @patch('bertopic.BERTopic.load')
    def test_load_model_success(self, mock_load):
        # Mock successful model loading
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        result = load_model()
        assert result == mock_model
        mock_load.assert_called_once_with("evstiars/bertopic-fake-news-detector")

    @patch('bertopic.BERTopic.load')
    def test_load_model_failure(self, mock_load):
        # Mock failure
        mock_load.side_effect = Exception("Load failed")

        with pytest.raises(Exception):
            load_model()
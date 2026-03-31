from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_sentence_model():
    """Lightweight stand-in for SentenceTransformer in API tests."""
    model = MagicMock()
    model.max_seq_length = 8
    model.tokenizer.model_max_length = 8

    def encode_mock(text, add_special_tokens=True):
        # One token per character so long strings exceed max_seq_length
        n = len(text) if text else 1
        return list(range(max(1, n)))

    model.tokenizer.encode = encode_mock
    model.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    return model


@pytest.fixture
def client(mock_sentence_model):
    with patch("st_service.main.load_model", return_value=mock_sentence_model):
        from st_service.main import app

        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_encode_raises(mock_sentence_model):
    mock_sentence_model.encode.side_effect = RuntimeError("encode failed")
    with patch("st_service.main.load_model", return_value=mock_sentence_model):
        from st_service.main import app

        with TestClient(app) as c:
            yield c

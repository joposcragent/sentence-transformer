from unittest.mock import MagicMock

import numpy as np
import pytest

from st_service.embedding import encode_text, max_context_tokens, token_length


def test_token_length():
    m = MagicMock()
    m.tokenizer.encode = lambda text, add_special_tokens=True: [1, 2, 3]
    assert token_length(m, "abc") == 3


def test_max_context_tokens_prefers_minimum():
    m = MagicMock()
    m.max_seq_length = 100
    m.tokenizer.model_max_length = 50
    assert max_context_tokens(m) == 50


def test_max_context_tokens_fallback():
    class Tok:
        model_max_length = None

    class M:
        max_seq_length = None
        tokenizer = Tok()

    assert max_context_tokens(M()) == 512


def test_encode_text_1d():
    m = MagicMock()
    m.encode.return_value = np.array([0.5, 0.5], dtype=np.float32)
    out = encode_text(m, "x")
    assert list(out) == [0.5, 0.5]


def test_encode_text_2d():
    m = MagicMock()
    m.encode.return_value = np.array([[0.1, 0.2]], dtype=np.float32)
    out = encode_text(m, "x")
    assert list(out) == pytest.approx([0.1, 0.2])

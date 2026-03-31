import numpy as np
import pytest

from st_service.cosine import cosine_similarity_01


def test_cosine_endpoint_identical_vectors(client):
    r = client.post(
        "/sentence-transformer/vectors/cosine-similarity",
        json={"left": [1.0, 0.0], "right": [1.0, 0.0]},
    )
    assert r.status_code == 200
    data = r.json()
    assert "similarity" in data
    assert 0.0 <= data["similarity"] <= 1.0
    assert data["similarity"] == pytest.approx(1.0)


def test_cosine_endpoint_mismatch_dimensions(client):
    r = client.post(
        "/sentence-transformer/vectors/cosine-similarity",
        json={"left": [1.0, 0.0], "right": [1.0]},
    )
    assert r.status_code == 500
    assert r.text


def test_cosine_similarity_01_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    s = cosine_similarity_01(a, b)
    assert s == pytest.approx(0.0)


def test_cosine_similarity_01_same():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0])
    s = cosine_similarity_01(a, b)
    assert s == pytest.approx(1.0)


def test_cosine_similarity_01_shape_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity_01(np.array([1.0]), np.array([1.0, 2.0]))

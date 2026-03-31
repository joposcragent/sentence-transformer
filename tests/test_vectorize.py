import pytest


def test_vectorize_ok(client):
    r = client.post(
        "/sentence-transformer/text/vectorize",
        json={"text": "hi"},
    )
    assert r.status_code == 200
    assert r.json() == pytest.approx([0.1, 0.2, 0.3])


def test_vectorize_413(client):
    long_text = "x" * 100
    r = client.post(
        "/sentence-transformer/text/vectorize",
        json={"text": long_text},
    )
    assert r.status_code == 413


def test_vectorize_500(client_encode_raises):
    r = client_encode_raises.post(
        "/sentence-transformer/text/vectorize",
        json={"text": "ok"},
    )
    assert r.status_code == 500
    assert "encode failed" in r.text

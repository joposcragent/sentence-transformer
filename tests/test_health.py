def test_health_ok(client):
    r = client.get("/sentence-transformer/health")
    assert r.status_code == 200
    assert r.json() == {"status": "READY"}

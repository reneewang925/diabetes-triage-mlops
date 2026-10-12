from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "v0.2"}


def test_predict():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)


def test_predict_missing_s6_returns_422():
    payload = VALID_PAYLOAD.copy()
    del payload["s6"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    body = response.json()
    assert "detail" in body
    assert any("s6" in str(item.get("loc", [])) for item in body["detail"])
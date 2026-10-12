# Diabetes Triage ML Service

Virtual diabetes clinic triage service. Predicts short-term disease progression
using the scikit-learn diabetes regression dataset. The clinic uses the returned
score to prioritize follow-up calls.

## Model versions

| Version | Model | Notes |
|---|---|---|
| v0.1 | StandardScaler + LinearRegression | Baseline |
| v0.2 | StandardScaler + ElasticNetCV | 53.441 | Auto-tuned regularization via 5-fold CV |

See `CHANGELOG.md` for metric deltas.

## Endpoints

### `GET /health`

Returns service status and model version.

```json
{"status": "ok", "model_version": "v0.1"}
```

### `POST /predict`

Request body: 10 diabetes features.

```json
{
  "age": 0.02,
  "sex": -0.044,
  "bmi": 0.06,
  "bp": -0.03,
  "s1": -0.02,
  "s2": 0.03,
  "s3": -0.02,
  "s4": 0.02,
  "s5": 0.02,
  "s6": -0.001
}
```

Response:

```json
{"prediction": 197.05843503852105}
```

### Error response (422)

Missing or invalid fields return HTTP 422 with a JSON body.
The `loc` field identifies which input is missing or invalid.

Example: missing `s6`.

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "s6"],
      "msg": "Field required",
      "input": {
        "age": 0.02,
        "sex": -0.044,
        "bmi": 0.06,
        "bp": -0.03,
        "s1": -0.02,
        "s2": 0.03,
        "s3": -0.02,
        "s4": 0.02,
        "s5": 0.02
      }
    }
  ]
}
```

## Local setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python training/train.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Test:

```bash
curl http://localhost:8000/health
python -m pytest -q
```

## Docker

Build (model must be trained first):

```bash
python training/train.py
docker build -t diabetes-api:v0.1 .
docker run -it -p 80:80 diabetes-api:v0.1
```

If port 80 is busy locally, use `-p 8080:80` and test against `localhost:8080`.

## Reproducibility

- Random seed: 42
- Pinned dependencies in `requirements.txt`
- Metrics saved to `models/<version>/metrics.json`
- CI retrains and rebuilds deterministically on every push and tag
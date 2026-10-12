# Diabetes Triage ML Service

Virtual diabetes clinic triage service. Predicts short-term disease progression
using the scikit-learn diabetes regression dataset. The clinic uses the returned
score to prioritize follow-up calls.

## Model versions

| Version | Model | Notes |
|---|---|---|
| v0.1 | StandardScaler + LinearRegression | Baseline |
| v0.2 | TBD | Improvement over v0.1 |

See `CHANGELOG.md` for metric deltas.

## Endpoints

### `GET /health`

Returns service status and model version.

```json
{"status": "ok", "model_version": "v0.1"}
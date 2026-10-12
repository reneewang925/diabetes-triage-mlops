# Changelog

## v0.2

### Changed

- Replaced `LinearRegression` with `ElasticNetCV`.
- `alpha` and `l1_ratio` are now selected automatically via 5-fold cross-validation
  on the training split (`l1_ratio` searched over `[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]`).
- Kept `StandardScaler` in the pipeline.

### Metrics

| Version | Model | RMSE (test split) |
|---|---|---|
| v0.1 | StandardScaler + LinearRegression | 53.8534 |
| v0.2 | StandardScaler + ElasticNetCV | 53.4415 |

Delta: **-0.4119 RMSE** (≈ -0.77%).

### Rationale

The diabetes regression dataset has 10 correlated features. Ordinary least
squares (LinearRegression) can overfit this kind of correlated input.
ElasticNet combines L1 and L2 penalties, which shrinks and selects coefficients,
reducing variance on the held-out test split. The regularization strength
`alpha` and the L1/L2 mix `l1_ratio` are chosen by cross-validation on the
training data, so the test split is never used for tuning. This is why v0.2
is preferred over v0.1.

## v0.1

- Model: StandardScaler + LinearRegression
- RMSE: 53.8534
- Random seed: 42
- Baseline release.
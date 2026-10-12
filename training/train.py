import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MODEL_VERSION = "v0.1"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / MODEL_VERSION
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    metrics = {
        "model_version": MODEL_VERSION,
        "rmse": rmse,
        "random_state": RANDOM_STATE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    joblib.dump(pipeline, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Saved model to {MODEL_DIR / 'model.joblib'}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
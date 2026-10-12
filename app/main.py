from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_VERSION = "v0.2"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / MODEL_VERSION / "model.joblib"

model = joblib.load(MODEL_PATH)

FEATURE_ORDER = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

app = FastAPI(title="Diabetes Triage API", version=MODEL_VERSION)


class PatientFeatures(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(features: PatientFeatures):
    data = pd.DataFrame([features.model_dump()], columns=FEATURE_ORDER)
    pred = model.predict(data)[0]
    return {"prediction": float(pred)}
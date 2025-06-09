import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow CORS for local React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all models from models directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
model_files = {
    'logistic_regression': 'logistic_regression.joblib',
    'random_forest': 'random_forest.joblib',
    'xgboost': 'xgboost.joblib',
    'mlp_classifier': 'mlp_classifier.joblib',
}
models = {}
for name, fname in model_files.items():
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path):
        models[name] = joblib.load(path)

# Example: get feature names from one model (assume all models use same features)
if models:
    example_model = next(iter(models.values()))
    try:
        feature_names = example_model.feature_names_in_.tolist()
    except AttributeError:
        feature_names = []
else:
    feature_names = []

class PredictRequest(BaseModel):
    model: str
    features: dict

@app.get("/models")
def get_models():
    return {"models": list(models.keys()), "features": feature_names}

@app.post("/predict")
def predict(req: PredictRequest):
    if req.model not in models:
        raise HTTPException(status_code=400, detail="Model not found.")
    model = models[req.model]
    # Ensure features are in correct order
    if feature_names is None:
        raise HTTPException(status_code=500, detail="Feature names not available.")
    try:
        X = np.array([[req.features[f] for f in feature_names]])
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")
    pred = model.predict(X)[0]
    prob = float(model.predict_proba(X)[0][1]) if hasattr(model, 'predict_proba') else None
    return {"prediction": int(pred), "probability": prob} 
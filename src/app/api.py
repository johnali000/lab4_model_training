# src/app/api.py
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Explicit request schema for Iris dataset (4 features)
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def create_app(model_path: str = "models/iris_model.pkl"):
    """
    Creates a FastAPI app that serves predictions for the Iris model.

    Example values that commonly predict each class:
      - setosa:     5.1, 3.5, 1.4, 0.2
      - versicolor: 6.0, 2.9, 4.5, 1.5
      - virginica:  6.9, 3.1, 5.4, 2.1
    """
    # Helpful guard so students get a clear error if they forgot to train first
    if not Path(model_path).exists():
        raise RuntimeError(
            f"Model file not found at '{model_path}'. "
            "Train the model first (run the DAG or scripts/train_model.py)."
        )

    model = joblib.load(model_path)
    app = FastAPI(title="Iris Model API")

    # Map numeric predictions to class names
    target_names = {0: "setosa", 1: "versicolor", 2: "virginica"}

    @app.get("/")
    def root():
        return {
            "message": "Iris model is ready for inference!",
            "classes": target_names,
        }

    @app.post("/predict")
    def predict(request: IrisRequest):
        # Convert request into the correct shape (1 x 4)
        X = np.array([
            [request.sepal_length, request.sepal_width,
             request.petal_length, request.petal_width]
        ])
        try:
            idx = int(model.predict(X)[0])
        except Exception as e:
            # Surface any shape/validation issues as a 400 instead of a 500
            raise HTTPException(status_code=400, detail=str(e))
        return {"prediction": target_names[idx], "class_index": idx}
    @app.get("/model/info")
    def get_model_info():
        metadata_path = Path('model/metadata.json')
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail='metadata.json not found')

        try:
            with metadata_path.open('r') as f:
                metadata = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'error reading metadata.json: {e}')
        return metadata

    # return the FastAPI app
    return app

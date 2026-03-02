from fastapi import FastAPI
from pydantic import BaseModel

from api.inference import CaloriePredictor

app = FastAPI(title="Calorie Burn Predictor API")

predictor = CaloriePredictor()


class PredictRequest(BaseModel):
    gender: str
    age: int
    height: float
    weight: float
    duration: float
    heart_rate: float
    body_temp: float


class PredictResponse(BaseModel):
    calories: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    result = predictor.predict(req.dict())
    return {"calories": round(result, 2)}
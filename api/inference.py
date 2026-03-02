import joblib
import pandas as pd

MODEL_PATH = "models/calorie_model.joblib"


class CaloriePredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, payload: dict) -> float:
        # Map API input to training feature format
        row = {
            "Gender": payload["gender"].strip().lower(),
            "Age": payload["age"],
            "Height": payload["height"],
            "Weight": payload["weight"],
            "Duration": payload["duration"],
            "Heart_Rate": payload["heart_rate"],
            "Body_Temp": payload["body_temp"],
        }

        df = pd.DataFrame([row])
        prediction = float(self.model.predict(df)[0])
        return prediction
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import load_calorie_data, FEATURES, TARGET
from src.model import make_model

DATA_PATH = "data/calories.csv"
MODEL_OUT = "models/calorie_model.joblib"


def train_and_eval():
    df = load_calorie_data(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = make_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # sklearn version-safe RMSE
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": rmse,
        "R2": r2_score(y_test, preds),
    }

    return model, metrics


def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_OUT)


def main():
    print("Training model...")
    model, metrics = train_and_eval()

    print("\nResults on test split:")
    print(f"  MAE:  {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  R^2:  {metrics['R2']:.3f}")

    save_model(model)
    print(f"\nSaved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
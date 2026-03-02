from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from src.features import FEATURES

CAT_COLS = ["Gender"]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]


@dataclass(frozen=True)
class TrainConfig:
    random_state: int = 42
    n_estimators: int = 300


def make_model(cfg: TrainConfig = TrainConfig()) -> Pipeline:
    """Create a preprocessing + model pipeline."""
    preprocess = ColumnTransformer(
        transformers=[
            ("gender", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("nums", "passthrough", NUM_COLS),
        ]
    )

    regressor = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", regressor),
        ]
    )
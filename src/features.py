import pandas as pd

FEATURES = [
    "Gender",
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp",
]

TARGET = "Calories"


def load_calorie_data(csv_path: str) -> pd.DataFrame:
    """Load and lightly clean the calories dataset."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]  # just in case

    _validate_columns(df)
    df = _basic_clean(df)

    return df


def _validate_columns(df: pd.DataFrame) -> None:
    needed = set(FEATURES + [TARGET])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}\nFound: {list(df.columns)}")


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing values in the columns we care about
    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    # Normalize gender labels
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()
    df["Gender"] = df["Gender"].replace({"m": "male", "f": "female"})

    # Keep only known values (helps avoid weird strings)
    df = df[df["Gender"].isin(["male", "female"])].copy()

    return df
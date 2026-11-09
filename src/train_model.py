"""Train and evaluate the COVID-19 mortality regression model.

Run from any working directory with: python src/train_model.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "covid_merged.csv"
MODEL_FILE = ROOT / "models" / "covid_death_predictor.pkl"
FEATURES = ["Confirmed", "Active", "New cases", "Hospital_beds_per_1000"]
TARGET = "Deaths"


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found at {DATA_FILE}. "
            "Run python src/merge_data.py first."
        )

    df = pd.read_csv(DATA_FILE)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Training dataset is missing columns: {', '.join(missing)}")

    clean = df[required].apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[(clean[FEATURES] >= 0).all(axis=1) & (clean[TARGET] >= 0)]

    if len(clean) < 10:
        raise ValueError(
            f"Only {len(clean)} valid rows are available. At least 10 are required."
        )

    X = clean[FEATURES]
    y = clean[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("\nModel evaluation on held-out rows")
    print(f"R²: {r2_score(y_test, predictions):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.4f}")

    folds = min(5, len(clean))
    if folds >= 2:
        scores = cross_val_score(model, X, y, cv=folds, scoring="r2", n_jobs=-1)
        print(f"Cross-validation R² ({folds} folds): {scores.mean():.4f}")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved model to: {MODEL_FILE}")


if __name__ == "__main__":
    main()

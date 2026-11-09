"""Run a sample prediction with the repository's saved model."""
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "covid_death_predictor.pkl"
FEATURES = ["Confirmed", "Active", "New cases", "Hospital_beds_per_1000"]


def main() -> None:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_FILE}")

    model = joblib.load(MODEL_FILE)
    sample = pd.DataFrame(
        [{
            "Confirmed": 100000,
            "Active": 5000,
            "New cases": 200,
            "Hospital_beds_per_1000": 3.0,
        }],
        columns=FEATURES,
    )
    estimate = max(0.0, float(model.predict(sample)[0]))
    print(f"Estimated reported deaths: {estimate:.0f}")


if __name__ == "__main__":
    main()

"""Merge the repository's COVID-19 and hospital-bed CSV snapshots."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
COVID_FILE = DATA_DIR / "country_wise_latest.csv"
BEDS_FILE = DATA_DIR / "API_SH.MED.BEDS.ZS_DS2_en_csv_v2_10288.csv"
OUTPUT_FILE = DATA_DIR / "covid_merged.csv"


def main() -> None:
    for path in (COVID_FILE, BEDS_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required source file not found: {path}")

    covid = pd.read_csv(COVID_FILE)
    beds = pd.read_csv(BEDS_FILE, skiprows=4)

    required_covid = {"Country/Region"}
    missing = required_covid.difference(covid.columns)
    if missing:
        raise ValueError(f"COVID dataset is missing columns: {', '.join(sorted(missing))}")

    if not {"Country Name"}.issubset(beds.columns):
        raise ValueError("Hospital-bed dataset must contain a 'Country Name' column.")

    year_columns = [column for column in beds.columns if str(column).isdigit()]
    if not year_columns:
        raise ValueError("No year columns were found in the hospital-bed dataset.")

    latest_year = max(year_columns, key=lambda value: int(value))
    bed_values = beds[["Country Name", latest_year]].rename(
        columns={
            "Country Name": "Country/Region",
            latest_year: "Hospital_beds_per_1000",
        }
    )
    bed_values["Hospital_beds_per_1000"] = pd.to_numeric(
        bed_values["Hospital_beds_per_1000"], errors="coerce"
    )

    merged = covid.merge(bed_values, on="Country/Region", how="left")
    merged["Hospital_beds_per_1000"] = merged["Hospital_beds_per_1000"].fillna(0)
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Merged {len(merged):,} country rows into {OUTPUT_FILE}")
    print(
        "Note: missing hospital-bed observations are zero-filled in this legacy "
        "dataset and should not be interpreted as measured zero availability."
    )


if __name__ == "__main__":
    main()

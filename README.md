# COVID-19 Mortality Prediction Lab

An interactive machine learning project for exploring historical COVID-19 indicators and estimating reported mortality counts with a trained Random Forest regression model.

<p align="center">
  <a href="https://mortality-prediction-covid.streamlit.app/"><strong>Open the live application</strong></a>
  &nbsp; · &nbsp;
  <a href="https://github.com/SudharsanamRK/covid19-death-prediction"><strong>View source code</strong></a>
</p>

## Project overview

The application brings together a trained regression model and a Streamlit analytics workspace. Users can enter one observation, process a CSV file, inspect model feature importance, compare country-level historical indicators and explore illustrative growth scenarios.

This is an educational data science project. It is not a clinical tool, an official surveillance platform or a validated system for making public health decisions.

## Live demo

**[mortality-prediction-covid.streamlit.app](https://mortality-prediction-covid.streamlit.app/)**

The application is hosted on Streamlit Community Cloud. The country comparison page reads the historical snapshot stored in this repository and is not a live data feed.

## Features

| Workspace | What it does |
| --- | --- |
| Dashboard | Project overview, quick navigation and explanation of the workflow |
| Single Prediction | Estimates deaths for one set of four input indicators |
| Batch Prediction | Validates CSV rows, processes valid records and exports results |
| Model Insights | Shows estimator information and fitted feature-importance scores |
| Country Comparison | Filters and compares historical country-level indicators |
| Trend Simulation | Explores user-defined illustrative growth scenarios |
| About & Methodology | Documents the data, model approach and responsible-use limitations |

## How the model works

The saved Random Forest regression model accepts four input features:

1. `Confirmed`: reported confirmed cases.
2. `Active`: reported active cases.
3. `New cases`: newly reported cases in the observation.
4. `Hospital_beds_per_1000`: hospital beds per 1,000 people in the source data.

The model was trained to estimate the reported `Deaths` target. The application loads the committed model artifact from `models/covid_death_predictor.pkl`. It does not train a new model when a user opens the dashboard.

The batch workflow applies the same model to each valid row in an uploaded CSV. Invalid rows are marked for review rather than silently included.

## Technology stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn and Joblib
- Plotly
- CSV datasets

## Repository structure

```text
covid19-death-prediction/
├── .github/
│   └── workflows/
│       └── python-checks.yml
├── .streamlit/
│   └── config.toml
├── .devcontainer/
│   └── devcontainer.json
├── assets/
│   └── icons/
├── data/
│   ├── country_wise_latest.csv
│   ├── covid_merged.csv
│   └── API_SH.MED.BEDS.ZS_DS2_en_csv_v2_10288.csv
├── models/
│   └── covid_death_predictor.pkl
├── src/
│   ├── app.py
│   ├── merge_data.py
│   ├── predict.py
│   └── train_model.py
├── requirements.txt
└── README.md
```

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/SudharsanamRK/covid19-death-prediction.git
cd covid19-death-prediction
```

### 2. Create and activate a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the application

Run this command from the repository root:

```bash
streamlit run src/app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Rebuild the model (optional)

The trained model is already included, so retraining is not required to run the application.

To rebuild it from the merged dataset:

```bash
python src/train_model.py
```

To rebuild the merged dataset from the source CSV files:

```bash
python src/merge_data.py
```

These scripts use paths relative to the repository, so they can run from different machines without editing local directory names.

## Data and methodology notes

- The repository contains static CSV snapshots. Their values should be interpreted in the context of their original collection and reporting dates.
- Country names and reporting definitions can differ between source datasets.
- Missing hospital-bed values are represented as zero in the existing merged data. Zero-filled values are not equivalent to measured zero bed availability.
- Feature importance describes how features were used by the fitted forest. It does not establish causation or explain the direction of an individual prediction.
- A prediction is an estimate learned from historical training data. It is not a forecast guarantee, a death count from an official authority, or an individual risk assessment.
- Model performance should be re-evaluated on representative held-out and time-separated data before any real-world use.

## Responsible use

This project is intended for learning, demonstration and exploratory analysis. Do not use its output for diagnosis, treatment, hospital planning, resource allocation or public policy decisions.

## License

Distributed under the MIT License. See [LICENSE](LICENSE).

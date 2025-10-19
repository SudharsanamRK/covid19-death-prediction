## COVID-19 Mortality Prediction System
  A machine learning application that predicts the number of COVID-19 deaths based on country-level epidemiological and healthcare infrastructure data. The project integrates real-world COVID-19 case statistics with hospital bed availability data to enhance prediction accuracy.

#### Overview
  This project uses a Random Forest Regressor model trained on COVID-19 country-wise data and healthcare indicators. The model provides an estimated number of deaths for a given set of inputs such as confirmed cases, active cases, new cases, and hospital beds per 1,000 people.
  A user-friendly web interface is built using Streamlit for real-time prediction and visualization.

#### Features
1. Data preprocessing and merging from multiple global datasets
2. Machine learning model training and evaluation
3. Integration of healthcare capacity indicators (hospital beds per 1,000 population)
4. Interactive Streamlit-based web dashboard for mortality prediction
5. Model persistence using Joblib

#### Live Demo
> Access the deployed app here:
https://covid19-death-prediction.streamlit.app/

#### Technologies Used
1. Python 3.12
2. Pandas, NumPy, Scikit-learn
3. Streamlit for UI
4. Joblib for model serialization

#### Future Improvements
1. Incorporate vaccination and demographic indicators
2. Add time-series forecasting capability
3. Integrate real-time data API sources

#### License
This project is open source and available under the MIT License.

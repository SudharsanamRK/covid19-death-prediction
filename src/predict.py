import pandas as pd
import joblib

# Load model
model = joblib.load("../models/covid_death_predictor.pkl")

# Example input
input_data = pd.DataFrame([{
    'Confirmed': 100000,
    'Active': 5000,
    'New cases': 200,
    'Hospital_beds_per_1000': 3
}])

prediction = model.predict(input_data)
print(f"Predicted COVID-19 Deaths: {prediction[0]:.0f}")

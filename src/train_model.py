import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import numpy as np

# Paths
DATA_DIR = r"C:\Users\SEC\OneDrive\Desktop\covid-19-death-prediction\data"
MODEL_DIR = r"C:\Users\SEC\OneDrive\Desktop\covid-19-death-prediction\models"
MERGED_FILE = os.path.join(DATA_DIR, "covid_merged.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "covid_death_predictor.pkl")

# Load merged dataset
df = pd.read_csv(MERGED_FILE)

# Features and target
features = ['Confirmed', 'Active', 'New cases', 'Hospital_beds_per_1000']
X = df[features].fillna(0)
y = df['Deaths'].fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()

print("\nModel Evaluation:")
print(f"R² Score: {r2}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"Cross-validation R²: {cv_r2}")

# Save model
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_FILE)
print(f"\n✅ Model trained and saved as '{MODEL_FILE}'")

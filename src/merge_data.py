import pandas as pd
import os

# Paths
DATA_DIR = r"C:\Users\SEC\OneDrive\Desktop\covid-19-death-prediction\data"
COVID_FILE = os.path.join(DATA_DIR, "country_wise_latest.csv")
BEDS_FILE = os.path.join(DATA_DIR, "API_SH.MED.BEDS.ZS_DS2_en_csv_v2_10288.csv")
MERGED_FILE = os.path.join(DATA_DIR, "covid_merged.csv")

# Load COVID data
covid_df = pd.read_csv(COVID_FILE)
print("COVID data columns:", covid_df.columns)

# Load hospital beds data
beds_df = pd.read_csv(BEDS_FILE, skiprows=4)
print("Hospital beds data columns:", beds_df.columns)

# Keep only necessary columns: country + latest year
latest_year = beds_df.columns[-1]  # automatically pick last column (latest year)
beds_df = beds_df[['Country Name', latest_year]]
beds_df = beds_df.rename(columns={'Country Name': 'Country/Region', latest_year: 'Hospital_beds_per_1000'})
beds_df['Hospital_beds_per_1000'] = beds_df['Hospital_beds_per_1000'].fillna(0)

# Merge datasets
merged_df = covid_df.merge(beds_df, on='Country/Region', how='left')
merged_df['Hospital_beds_per_1000'] = merged_df['Hospital_beds_per_1000'].fillna(0)

# Save merged CSV
merged_df.to_csv(MERGED_FILE, index=False)
print("✅ Merged dataset saved as covid_merged.csv")

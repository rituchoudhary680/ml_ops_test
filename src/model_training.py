"""
ML Ops Test Script

Instructions:
- Build this script from scratch.
- Ensure all necessary steps are included as per the task requirements.
- Document your code clearly.
- Include your name and the date at the top of this file.

Applicant Name: [Your Name]
Date: [Current Date]

"""

# Your code starts here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import os

def train_model():
    # Load preprocessed data
    data = pd.read_csv('data/preprocessed_data.csv')

    # Define features and target
    X = data[['outside_weather_degree_celcius', 'RelativeHumidity', 'Temperature',
              'hour_of_day', 'day_of_week', 'temperature_humidity_index']]
    y = data['building_energy_consumption_kwh']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Ensure the directory exists
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    dump(model, os.path.join(model_dir, 'random_forest_model.joblib'))

if __name__ == "__main__":
    train_model()

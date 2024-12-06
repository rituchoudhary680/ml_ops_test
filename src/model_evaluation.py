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
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load

def evaluate_model():
    # Load preprocessed data
    data = pd.read_csv('data/preprocessed_data.csv')

    # Define features and target
    X = data[['outside_weather_degree_celcius', 'RelativeHumidity', 'Temperature',
              'hour_of_day', 'day_of_week', 'temperature_humidity_index']]
    y = data['building_energy_consumption_kwh']

    # Load model
    model = load('model/random_forest_model.joblib')

    # Predict and evaluate
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

if __name__ == "__main__":
    evaluate_model()

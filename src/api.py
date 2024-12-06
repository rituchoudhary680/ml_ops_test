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
from fastapi import FastAPI
from joblib import load
import numpy as np
import pandas as pd
import logging
app = FastAPI()

# Load the trained model
model = load('model/random_forest_model.joblib')

@app.post("/predict/")
def predict(features: dict):
    logging.info(f"Received input: {features}")
    print (f"Received input: {features}")
    prediction = [None]
    try:
        feature_array = np.array([[
            features['outside_weather_degree_celcius'],
            features['RelativeHumidity'],
            features['Temperature'],
            features['hour_of_day'],
            features['day_of_week'],
            features['temperature_humidity_index']
        ]])
        prediction = model.predict(feature_array)
    except Exception as e:
        print (f"Error occured in mumpy method trying pandas, {e}")
        


    return {"predicted_energy_consumption_kwh": prediction[0]}

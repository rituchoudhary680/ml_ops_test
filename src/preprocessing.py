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
import numpy as np

def preprocess_data():
    # Load datasets
    energy_data = pd.read_csv('data/energy_data.csv')
    sensor_data = pd.read_csv('data/sensor_data.csv')
    weather_data = pd.read_csv('data/weather_data.csv')

    # Convert timestamps
    energy_data['timestamp'] = pd.to_datetime(energy_data['timestamp'])
    sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp'])
    weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
    sensor_data['timestamp'] = sensor_data['timestamp'].dt.tz_localize(None)
    # Remove timezone info and clean `tags_name`
    sensor_data['timestamp'] = sensor_data['timestamp'].dt.tz_localize(None)
    sensor_data['tags_name'] = sensor_data['tags_name'].str.strip("[]").str.strip("'\"")
    # Aggregate sensor data
    # Aggregate numeric columns with mean and non-numeric with first
    sensor_data_aggregated = sensor_data.groupby(pd.Grouper(key='timestamp', freq='10T')).agg(
        {
            'tags_name': 'first',   # Keep the first value for tags_name
            'site_id': 'first',     # Keep the first value for site_id
            'RelativeHumidity': 'mean',  # Average for numeric
            'Temperature': 'mean'        # Average for numeric
        }
    ).reset_index()

    # Merge datasets
    merged_data = pd.merge(energy_data, weather_data, on='timestamp', how='inner')
    merged_data = pd.merge(merged_data, sensor_data_aggregated, on='timestamp', how='left')
    merged_data['site_id'] = merged_data['site_id'].fillna(method='ffill').fillna(method='bfill')

    # Handle missing temperature and humidity using rolling mean for localized imputation
    for col in ['RelativeHumidity', 'Temperature']:
        # Rolling mean imputation
        merged_data[col] = merged_data[col].fillna(
            merged_data[col].rolling(window=10, min_periods=1, center=True).mean()
        )
        # Fill residual nulls with the overall column mean
        merged_data[col] = merged_data[col].fillna(merged_data[col].mean())

    # Ensure `tags_name` uses a similar forward/backward filling approach
    merged_data['tags_name'] = merged_data['tags_name'].fillna(method='ffill').fillna(method='bfill')

    # Fill missing values for analysis
    merged_data['RelativeHumidity'] = merged_data['RelativeHumidity'].fillna(method='ffill').fillna(method='bfill')
    merged_data['Temperature'] = merged_data['Temperature'].fillna(method='ffill').fillna(method='bfill')

    # Add derived time-based features
    merged_data['hour_of_day'] = merged_data['timestamp'].dt.hour
    merged_data['day_of_week'] = merged_data['timestamp'].dt.dayofweek
    merged_data['temperature_humidity_index'] = (
        0.8 * merged_data['Temperature'] +
        (merged_data['RelativeHumidity'] / 100) * (merged_data['Temperature'] - 14.4) + 46.4
    )

    # Re-index and fill missing timestamps(consistent time range)
    timestamp_range = pd.date_range(start=energy_data['timestamp'].min(), end=energy_data['timestamp'].max(), freq='10T')
    merged_data = merged_data.set_index('timestamp').reindex(timestamp_range).reset_index()
    merged_data.rename(columns={'index': 'timestamp'}, inplace=True)
    merged_data['RelativeHumidity'] = merged_data['RelativeHumidity'].fillna(method='ffill').fillna(method='bfill')
    merged_data['Temperature'] = merged_data['Temperature'].fillna(method='ffill').fillna(method='bfill')

    # Add derived features
    merged_data['hour_of_day'] = merged_data['timestamp'].dt.hour
    merged_data['day_of_week'] = merged_data['timestamp'].dt.dayofweek
    merged_data['temperature_humidity_index'] = (
        0.8 * merged_data['Temperature'] +
        (merged_data['RelativeHumidity'] / 100) * (merged_data['Temperature'] - 14.4) + 46.4
    )

    # Save preprocessed data
    merged_data.to_csv('data/preprocessed_data.csv', index=False)

if __name__ == "__main__":
    preprocess_data()

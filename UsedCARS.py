# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 18:27:45 2025

@author: Sonal
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Title of the app
st.title("Used Car Price Prediction")

# Sidebar for user input
st.sidebar.header("Input Features")
kilometer = st.sidebar.number_input("Kilometers Driven", min_value=0, step=1000)
model_year = st.sidebar.number_input("Model Year", min_value=1900, max_value=2025, step=1)
car_age = st.sidebar.number_input("Car Age", min_value=0, step=1)

# Categorical inputs
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"])
transmission = st.sidebar.selectbox("Transmission Type", ["Manual", "Automatic"])

# Map categorical inputs to encoded values
fuel_type_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2, "Electric": 3, "Hybrid": 4}
transmission_mapping = {"Manual": 0, "Automatic": 1}
fuel_type_encoded = fuel_type_mapping[fuel_type]
transmission_encoded = transmission_mapping[transmission]

# Load dataset
def load_data():
    df = pd.read_csv("usedCars.csv")  # Replace with the actual filename
    return df

# Preprocess dataset

def preprocess_data(df):
    # Process the DataFrame as before
    df = df.copy()
    df['Kilometer'] = df['Kilometer'].astype(str).str.replace(',', '').astype(float)
    df['Price'] = df['Price'].astype(str).str.replace(' Lakhs', '', regex=False).str.replace(',', '').astype(float)
    df['FuelType'] = df['FuelType'].map(fuel_type_mapping)
    df['TransmissionType'] = df['TransmissionType'].map(transmission_mapping)
    df['CarAge'] = 2025 - df['ModelYear']
    df['Price'] = df['Price'].fillna(df['Price'].median())
    df['Kilometer'] = df['Kilometer'].fillna(df['Kilometer'].median())
    df['CarAge'] = df['CarAge'].fillna(0)
    return df[['Kilometer', 'ModelYear', 'CarAge', 'FuelType', 'TransmissionType', 'Price']]

# Train Decision Tree model
def train_model(df):
    X = df[['Kilometer', 'ModelYear', 'CarAge', 'FuelType', 'TransmissionType']]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = DecisionTreeRegressor(max_depth=5, random_state=1)
    model.fit(X_train, y_train)
    return model

# Load and preprocess the data
data = load_data()
data = preprocess_data(data)

# Train the model
model = train_model(data)

# Predict button
if st.sidebar.button("Predict Price"):
    # Combine inputs into a feature array
    features = np.array([[kilometer, model_year, car_age, fuel_type_encoded, transmission_encoded]])

    # Predict using the model
    prediction = model.predict(features)[0]

    # Display the result
    st.subheader("Predicted Price")
    st.write(f"The estimated price of the car is: ₹{prediction:,.2f} Lakhs")
    
    
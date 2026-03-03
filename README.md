# 🚗 Used Car Price Prediction

A machine learning web application that predicts the price of used cars based on key features like mileage, model year, fuel type, and transmission type.

## 🔍 Problem Statement
Buying or selling a used car is often a guessing game. This project builds a predictive model to estimate fair market prices, helping both buyers and sellers make informed decisions.

## 🛠️ Tech Stack
- **Python** — Pandas, NumPy, Scikit-learn, XGBoost
- **Visualization** — Matplotlib, Seaborn
- **Deployment** — Streamlit

## 📊 Models Used
- Linear Regression
- Lasso & Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

## ⚙️ Features Used for Prediction
- Kilometers Driven
- Model Year / Car Age
- Fuel Type (Petrol, Diesel, CNG, Electric, Hybrid)
- Transmission Type (Manual / Automatic)

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run UsedCARS.py
```

## 📁 Project Structure
```
├── Used Car Prices.ipynb       # Full analysis and model building
├── UsedCARS.py                 # Streamlit web app
├── usedCars.csv                # Dataset
```

## 📌 Key Insights
- Car age and kilometers driven are the strongest predictors of price
- Diesel cars retain value better than petrol in the used market
- XGBoost and Random Forest outperformed linear models significantly

---
*Project completed as part of Data Science internship at Spinnaker Analytics LLC (2024–2025)*


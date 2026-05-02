import streamlit as st
import pandas as pd
import joblib
import os

st.title("Colombo Stock Exchange Prediction App")

# Load data
data_path = 'data/historical_data.csv'
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    stocks = df['Symbol'].unique()
else:
    st.error("Data not found. Please run data_collector.py first.")
    stocks = []

stock = st.selectbox("Select Stock", stocks)

if stock:
    stock_data = df[df['Symbol'] == stock]
    st.subheader(f"Historical Data for {stock}")
    st.line_chart(stock_data.set_index('Date')['Close'])

    # Load model
    model_path = f'models/{stock}_model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        last_row = stock_data.iloc[-1]
        features = [[last_row['Close'], last_row['Volume']]]
        prediction = model.predict(features)[0]
        st.subheader("Prediction")
        st.write(f"Predicted next day's close price: {prediction:.2f}")
    else:
        st.error("Model not found. Please run train_model.py first.")
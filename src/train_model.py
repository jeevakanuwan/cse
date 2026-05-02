import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_model():
    # Load data
    data_path = 'data/historical_data.csv'
    if not os.path.exists(data_path):
        print("Data not found. Run data_collector.py first.")
        return

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date'])

    # For simplicity, predict next day's close for each stock
    models = {}
    for symbol in df['Symbol'].unique():
        stock_data = df[df['Symbol'] == symbol].copy()
        stock_data['Next_Close'] = stock_data['Close'].shift(-1)
        stock_data = stock_data.dropna()

        # Features: Close, Volume
        X = stock_data[['Close', 'Volume']]
        y = stock_data['Next_Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Model for {symbol}: MSE = {mse}")

        models[symbol] = model

        # Save model
        joblib.dump(model, f'models/{symbol}_model.pkl')

    print("Models trained and saved.")

if __name__ == "__main__":
    train_model()
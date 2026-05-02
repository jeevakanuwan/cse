import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data():
    # Generate sample historical data for a few CSE stocks
    stocks = ['JKH.N0000', 'COMB.N0000', 'HNB.N0000']  # Sample stock codes
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')

    data = []
    for stock in stocks:
        # Generate random prices with trend
        base_price = np.random.uniform(50, 500)
        prices = []
        current_price = base_price
        for _ in dates:
            change = np.random.normal(0, 0.02)  # 2% volatility
            current_price *= (1 + change)
            prices.append(current_price)

        df = pd.DataFrame({
            'Date': dates,
            'Symbol': stock,
            'Open': prices,
            'High': [p * np.random.uniform(1, 1.05) for p in prices],
            'Low': [p * np.random.uniform(0.95, 1) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000, 100000, len(dates))
        })
        data.append(df)

    combined_df = pd.concat(data, ignore_index=True)
    combined_df.to_csv('data/historical_data.csv', index=False)
    print("Sample historical data generated and saved to data/historical_data.csv")

if __name__ == "__main__":
    generate_sample_data()
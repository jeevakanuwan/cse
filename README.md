# Colombo Stock Exchange Prediction App

This application collects historical trading data from the Colombo Stock Exchange (CSE) and uses machine learning models to predict future trading trends.

## Features

- Data collection from CSE
- Historical data storage
- Predictive modeling
- Web interface for visualization

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the data collector:
   ```
   python src/data_collector.py
   ```

3. Train the model:
   ```
   python src/train_model.py
   ```

4. Launch the app:
   ```
   streamlit run src/app.py
   ```

## Project Structure

- `src/`: Source code
- `data/`: Historical data
- `models/`: Trained models
- `notebooks/`: Jupyter notebooks for analysis
- `tests/`: Unit tests
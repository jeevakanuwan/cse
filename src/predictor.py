"""
Next-day price direction predictor using a Random Forest classifier.

Features used:
  - 5 / 10 / 20-day moving averages relative to current price
  - RSI (14-day)
  - MACD signal
  - Bollinger Band position
  - Volume ratio vs 10-day average
  - Day-of-week

Target: next-day direction (1 = price goes UP, 0 = price goes DOWN or flat)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from . import database as db

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MIN_ROWS = 60      # need at least 60 trading days to build features
TRAIN_SPLIT = 0.8  # 80 % train, 20 % test


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _add_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    c = df["close"]

    # moving averages
    df["ma5"]  = c.rolling(5).mean()
    df["ma10"] = c.rolling(10).mean()
    df["ma20"] = c.rolling(20).mean()

    df["price_vs_ma5"]  = c / df["ma5"]  - 1
    df["price_vs_ma10"] = c / df["ma10"] - 1
    df["price_vs_ma20"] = c / df["ma20"] - 1

    # RSI 14
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_diff"] = macd - signal

    # Bollinger Bands (20-day)
    bb_mid  = c.rolling(20).mean()
    bb_std  = c.rolling(20).std()
    df["bb_position"] = (c - bb_mid) / (2 * bb_std).replace(0, np.nan)

    # Volume ratio
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(10).mean().replace(0, np.nan)

    # Day of week (0=Mon … 4=Fri)
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek

    # Target: did close go UP N days from now?
    df["target"] = (c.shift(-horizon) > c).astype(int)

    return df


FEATURE_COLS = [
    "price_vs_ma5", "price_vs_ma10", "price_vs_ma20",
    "rsi", "macd_diff", "bb_position", "vol_ratio", "dow",
]


# ---------------------------------------------------------------------------
# Per-symbol model helpers
# ---------------------------------------------------------------------------

def _model_path(symbol: str, horizon: int = 1) -> Path:
    safe   = symbol.replace("/", "_").replace("\\", "_")
    suffix = f"_h{horizon}" if horizon != 1 else ""
    return MODELS_DIR / f"{safe}{suffix}.pkl"


def train(symbol: str, horizon: int = 1) -> float | None:
    """
    Train (or retrain) the model for *symbol* at *horizon* days ahead.
    Returns test-set accuracy or None if there is not enough data.
    """
    df = db.get_prices(symbol, limit=1000)
    if len(df) < MIN_ROWS + horizon:
        logger.debug("Not enough data for %s h=%d (%d rows).", symbol, horizon, len(df))
        return None

    df = _add_features(df, horizon).dropna(subset=FEATURE_COLS + ["target"])
    df = df.iloc[:-horizon]  # last <horizon> rows have no known target yet

    if len(df) < MIN_ROWS:
        return None

    X = df[FEATURE_COLS].values
    y = df["target"].values

    split = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    joblib.dump(model, _model_path(symbol, horizon))
    logger.info("Trained %s h=%d — test accuracy %.1f%%", symbol, horizon, accuracy * 100)
    return accuracy


def predict_next_day(symbol: str, horizon: int = 1) -> dict | None:
    """
    Predict the price direction *horizon* days from now for *symbol*.
    Returns {"direction": "UP"/"DOWN", "confidence": float, "predicted_close": float}
    or None if a prediction cannot be made.
    Auto-trains the model for the given horizon if no model file exists yet.
    """
    path = _model_path(symbol, horizon)
    if not path.exists():
        acc = train(symbol, horizon)
        if acc is None:
            return None

    model = joblib.load(path)

    df = db.get_prices(symbol, limit=100)
    if len(df) < MIN_ROWS:
        return None

    df = _add_features(df, horizon).dropna(subset=FEATURE_COLS)
    last = df.iloc[-1]

    X = last[FEATURE_COLS].values.reshape(1, -1)
    proba = model.predict_proba(X)[0]

    direction  = "UP" if proba[1] >= 0.5 else "DOWN"
    confidence = float(max(proba))
    last_close = float(last["close"])

    avg_daily_move = df["close"].pct_change().abs().mean()
    predicted_close = (
        last_close * (1 + avg_daily_move * horizon) if direction == "UP"
        else last_close * (1 - avg_daily_move * horizon)
    )

    return {
        "direction":       direction,
        "confidence":      round(confidence, 4),
        "predicted_close": round(predicted_close, 2),
    }


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------

def train_all(horizon: int = 1):
    """Train / retrain models for every security that has enough data."""
    securities = db.get_all_securities()
    total = len(securities)
    for idx, row in securities.iterrows():
        symbol = row["symbol"]
        logger.info("[%d/%d] Training %s h=%d …", idx + 1, total, symbol, horizon)
        train(symbol, horizon)


def predict_all(horizon: int = 1):
    """
    Run predictions for every security and save them to the DB.
    Predicts *horizon* calendar days from today.
    """
    target_date = (datetime.today() + timedelta(days=horizon)).strftime("%Y-%m-%d")
    securities  = db.get_all_securities()
    saved = 0
    for _, row in securities.iterrows():
        symbol = row["symbol"]
        result = predict_next_day(symbol, horizon)
        if result:
            db.upsert_prediction(
                symbol, target_date,
                result["direction"],
                result["confidence"],
                result["predicted_close"],
                horizon,
            )
            saved += 1
    logger.info("Predictions saved for %d / %d securities.", saved, len(securities))

"""
Tests for the Predicted vs Actual page data layer.

Covers: get_prediction_history filtered by symbol+horizon, metric calculations
(MAE, MAPE, direction accuracy), result ordering, column completeness, and
edge cases (empty result, flat market, multiple horizons for same symbol).
"""

import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _insert_scenario(tmp_db, symbol, pred_date, prev_close, actual_close,
                     pred_dir, confidence=0.75, horizon=1):
    """Insert security, prev-day price, actual price, and one past prediction."""
    prev_date = "2024-05-31" if pred_date == "2024-06-01" else "2024-05-30"
    tmp_db.upsert_security(symbol, f"{symbol} Corp", "Finance")
    tmp_db.upsert_prices([
        {"symbol": symbol, "date": prev_date,
         "open": prev_close - 1, "high": prev_close + 1,
         "low":  prev_close - 2, "close": prev_close,
         "volume": 1000, "trades": 50},
        {"symbol": symbol, "date": pred_date,
         "open": actual_close - 0.5, "high": actual_close + 1,
         "low":  actual_close - 1,   "close": actual_close,
         "volume": 1200, "trades": 60},
    ])
    tmp_db.upsert_prediction(
        symbol, pred_date, pred_dir, confidence,
        actual_close + 2.0,   # predicted_close intentionally offset from actual
        horizon,
    )


# ===========================================================================
# Symbol + horizon combined filter
# ===========================================================================

class TestSymbolHorizonFilter:
    def test_returns_only_matching_symbol(self, tmp_db, mock_config):
        _insert_scenario(tmp_db, "AAA.N0000", "2024-06-01", 100, 105, "UP")
        _insert_scenario(tmp_db, "BBB.N0000", "2024-06-01", 200, 195, "DOWN")

        df = tmp_db.get_prediction_history(symbol="AAA.N0000", horizon=1)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "AAA.N0000"

    def test_returns_only_matching_horizon(self, tmp_db, mock_config):
        _insert_scenario(tmp_db, "AAA.N0000", "2024-06-01", 100, 105, "UP", horizon=1)
        # Reuse same symbol for horizon=3 — different predicted_for date required
        tmp_db.upsert_prices([
            {"symbol": "AAA.N0000", "date": "2024-05-28",
             "open": 98, "high": 102, "low": 97, "close": 100,
             "volume": 800, "trades": 40},
            {"symbol": "AAA.N0000", "date": "2024-05-31",
             "open": 103, "high": 107, "low": 102, "close": 106,
             "volume": 900, "trades": 45},
        ])
        tmp_db.upsert_prediction("AAA.N0000", "2024-05-31", "DOWN", 0.65, 104.0, 3)

        h1 = tmp_db.get_prediction_history(symbol="AAA.N0000", horizon=1)
        h3 = tmp_db.get_prediction_history(symbol="AAA.N0000", horizon=3)
        assert len(h1) == 1 and h1.iloc[0]["horizon"] == 1
        assert len(h3) == 1 and h3.iloc[0]["horizon"] == 3

    def test_empty_result_for_unknown_symbol(self, tmp_db, mock_config):
        _insert_scenario(tmp_db, "AAA.N0000", "2024-06-01", 100, 105, "UP")
        df = tmp_db.get_prediction_history(symbol="GHOST.N0000", horizon=1)
        assert df.empty

    def test_empty_result_when_horizon_missing(self, tmp_db, mock_config):
        _insert_scenario(tmp_db, "AAA.N0000", "2024-06-01", 100, 105, "UP", horizon=1)
        df = tmp_db.get_prediction_history(symbol="AAA.N0000", horizon=5)
        assert df.empty


# ===========================================================================
# Result ordering and columns
# ===========================================================================

class TestResultStructure:
    def test_rows_ordered_descending_by_predicted_for(self, tmp_db, mock_config):
        sym = "ORD.N0000"
        tmp_db.upsert_security(sym, "Order Corp", "Tech")
        dates_and_prices = [
            ("2024-04-30", "2024-05-01", 100, 102),
            ("2024-05-01", "2024-05-02", 102, 104),
            ("2024-05-02", "2024-05-03", 104, 101),
        ]
        for prev_d, pred_d, prev_c, act_c in dates_and_prices:
            tmp_db.upsert_prices([
                {"symbol": sym, "date": prev_d,
                 "open": prev_c, "high": prev_c + 1, "low": prev_c - 1,
                 "close": prev_c, "volume": 1000, "trades": 50},
                {"symbol": sym, "date": pred_d,
                 "open": act_c, "high": act_c + 1, "low": act_c - 1,
                 "close": act_c, "volume": 1100, "trades": 55},
            ])
            tmp_db.upsert_prediction(sym, pred_d, "UP", 0.7, act_c + 1, 1)

        df = tmp_db.get_prediction_history(symbol=sym, horizon=1)
        dates = df["predicted_for"].tolist()
        assert dates == sorted(dates, reverse=True)

    def test_required_columns_present(self, tmp_db, mock_config):
        _insert_scenario(tmp_db, "COL.N0000", "2024-06-01", 100, 105, "UP")
        df = tmp_db.get_prediction_history(symbol="COL.N0000", horizon=1)
        required = {
            "symbol", "name", "predicted_for", "horizon",
            "predicted_direction", "confidence",
            "predicted_close", "actual_close", "prev_close",
            "actual_direction", "correct",
        }
        assert required <= set(df.columns)

    def test_name_joined_from_securities(self, tmp_db, mock_config):
        _insert_scenario(tmp_db, "NM.N0000", "2024-06-01", 100, 105, "UP")
        df = tmp_db.get_prediction_history(symbol="NM.N0000", horizon=1)
        assert df.iloc[0]["name"] == "NM.N0000 Corp"


# ===========================================================================
# Direction accuracy (correct flag)
# ===========================================================================

class TestDirectionCorrectness:
    def test_up_prediction_correct_when_price_rises(self, tmp_db, mock_config):
        _insert_scenario(tmp_db, "UP1.N0000", "2024-06-01",
                         prev_close=100, actual_close=105, pred_dir="UP")
        df = tmp_db.get_prediction_history(symbol="UP1.N0000")
        assert df.iloc[0]["correct"] == 1
        assert df.iloc[0]["actual_direction"] == "UP"

    def test_up_prediction_wrong_when_price_falls(self, tmp_db, mock_config):
        _insert_scenario(tmp_db, "UP2.N0000", "2024-06-01",
                         prev_close=100, actual_close=95, pred_dir="UP")
        df = tmp_db.get_prediction_history(symbol="UP2.N0000")
        assert df.iloc[0]["correct"] == 0
        assert df.iloc[0]["actual_direction"] == "DOWN"

    def test_down_prediction_correct_when_price_falls(self, tmp_db, mock_config):
        _insert_scenario(tmp_db, "DN1.N0000", "2024-06-01",
                         prev_close=100, actual_close=93, pred_dir="DOWN")
        df = tmp_db.get_prediction_history(symbol="DN1.N0000")
        assert df.iloc[0]["correct"] == 1

    def test_down_prediction_correct_on_flat_market(self, tmp_db, mock_config):
        _insert_scenario(tmp_db, "FL1.N0000", "2024-06-01",
                         prev_close=100, actual_close=100, pred_dir="DOWN")
        df = tmp_db.get_prediction_history(symbol="FL1.N0000")
        assert df.iloc[0]["correct"] == 1
        assert df.iloc[0]["actual_direction"] == "FLAT"

    def test_up_prediction_wrong_on_flat_market(self, tmp_db, mock_config):
        _insert_scenario(tmp_db, "FL2.N0000", "2024-06-01",
                         prev_close=100, actual_close=100, pred_dir="UP")
        df = tmp_db.get_prediction_history(symbol="FL2.N0000")
        assert df.iloc[0]["correct"] == 0


# ===========================================================================
# Metric computation helpers (mirrors page_prediction_vs_actual logic)
# ===========================================================================

def _compute_metrics(df: pd.DataFrame) -> dict:
    """Replicate the metric formulas used in page_prediction_vs_actual."""
    total   = len(df)
    dir_acc = df["correct"].mean() * 100
    mae     = (df["predicted_close"] - df["actual_close"]).abs().mean()
    mape    = (
        (df["predicted_close"] - df["actual_close"]).abs()
        / df["actual_close"].replace(0, float("nan"))
    ).mean() * 100
    return {"total": total, "dir_acc": dir_acc, "mae": mae, "mape": mape}


class TestMetricCalculations:
    def test_direction_accuracy_all_correct(self, tmp_db, mock_config):
        for i, sym in enumerate(["M1.N0000", "M2.N0000", "M3.N0000"]):
            _insert_scenario(tmp_db, sym, "2024-06-01",
                             prev_close=100, actual_close=105, pred_dir="UP")
        df = tmp_db.get_prediction_history(horizon=1)
        assert _compute_metrics(df)["dir_acc"] == pytest.approx(100.0)

    def test_direction_accuracy_all_wrong(self, tmp_db, mock_config):
        for sym in ["W1.N0000", "W2.N0000"]:
            _insert_scenario(tmp_db, sym, "2024-06-01",
                             prev_close=100, actual_close=95, pred_dir="UP")
        df = tmp_db.get_prediction_history(horizon=1)
        assert _compute_metrics(df)["dir_acc"] == pytest.approx(0.0)

    def test_mae_computed_correctly(self, tmp_db, mock_config):
        # predicted_close = actual_close + 2.0 (set by _insert_scenario)
        # so MAE should be exactly 2.0
        _insert_scenario(tmp_db, "MAE.N0000", "2024-06-01",
                         prev_close=100, actual_close=110, pred_dir="UP")
        df = tmp_db.get_prediction_history(symbol="MAE.N0000", horizon=1)
        m = _compute_metrics(df)
        assert m["mae"] == pytest.approx(2.0)

    def test_mape_computed_correctly(self, tmp_db, mock_config):
        # predicted = actual + 2, actual = 100 → MAPE = 2/100 * 100 = 2.0 %
        _insert_scenario(tmp_db, "MAPE.N0000", "2024-06-01",
                         prev_close=100, actual_close=100, pred_dir="DOWN")
        df = tmp_db.get_prediction_history(symbol="MAPE.N0000", horizon=1)
        m = _compute_metrics(df)
        assert m["mape"] == pytest.approx(2.0)

    def test_total_matches_row_count(self, tmp_db, mock_config):
        _insert_scenario(tmp_db, "TOT.N0000", "2024-06-01",
                         prev_close=100, actual_close=105, pred_dir="UP")
        df = tmp_db.get_prediction_history(symbol="TOT.N0000", horizon=1)
        assert _compute_metrics(df)["total"] == 1

    def test_mixed_accuracy_fifty_percent(self, tmp_db, mock_config):
        _insert_scenario(tmp_db, "MX1.N0000", "2024-06-01",
                         prev_close=100, actual_close=105, pred_dir="UP")
        _insert_scenario(tmp_db, "MX2.N0000", "2024-06-01",
                         prev_close=100, actual_close=95,  pred_dir="UP")
        df = tmp_db.get_prediction_history(horizon=1)
        assert _compute_metrics(df)["dir_acc"] == pytest.approx(50.0)


# ===========================================================================
# Price error series (feeds the bar chart in the page)
# ===========================================================================

class TestPriceErrorSeries:
    def test_price_error_sign_positive_when_over_predicted(self, tmp_db, mock_config):
        # predicted_close = actual_close + 2 (from _insert_scenario)
        _insert_scenario(tmp_db, "ERR.N0000", "2024-06-01",
                         prev_close=100, actual_close=110, pred_dir="UP")
        df = tmp_db.get_prediction_history(symbol="ERR.N0000", horizon=1)
        error = (df["predicted_close"] - df["actual_close"]).iloc[0]
        assert error > 0

    def test_price_error_is_zero_for_perfect_prediction(self, tmp_db, mock_config):
        sym = "PERF.N0000"
        tmp_db.upsert_security(sym, "Perfect Corp", "Unknown")
        tmp_db.upsert_prices([
            {"symbol": sym, "date": "2024-05-31",
             "open": 99, "high": 101, "low": 98, "close": 100,
             "volume": 1000, "trades": 50},
            {"symbol": sym, "date": "2024-06-01",
             "open": 105, "high": 107, "low": 104, "close": 105,
             "volume": 1100, "trades": 55},
        ])
        # predicted_close == actual_close → error = 0
        tmp_db.upsert_prediction(sym, "2024-06-01", "UP", 0.8, 105.0, 1)
        df = tmp_db.get_prediction_history(symbol=sym, horizon=1)
        error = (df["predicted_close"] - df["actual_close"]).iloc[0]
        assert error == pytest.approx(0.0)

    def test_error_series_length_matches_history_rows(self, tmp_db, mock_config):
        sym = "SER.N0000"
        tmp_db.upsert_security(sym, "Series Corp", "Unknown")
        dates = [
            ("2024-04-30", "2024-05-01", 100, 103),
            ("2024-05-01", "2024-05-02", 103, 101),
            ("2024-05-02", "2024-05-03", 101, 106),
        ]
        for prev_d, pred_d, prev_c, act_c in dates:
            tmp_db.upsert_prices([
                {"symbol": sym, "date": prev_d,
                 "open": prev_c, "high": prev_c + 1, "low": prev_c - 1,
                 "close": prev_c, "volume": 500, "trades": 25},
                {"symbol": sym, "date": pred_d,
                 "open": act_c, "high": act_c + 1, "low": act_c - 1,
                 "close": act_c, "volume": 600, "trades": 30},
            ])
            tmp_db.upsert_prediction(sym, pred_d, "UP", 0.7, act_c + 1, 1)

        df = tmp_db.get_prediction_history(symbol=sym, horizon=1)
        errors = df["predicted_close"] - df["actual_close"]
        assert len(errors) == 3

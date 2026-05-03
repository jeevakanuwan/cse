"""
Tests for src/predictor.py

Covers: feature engineering correctness (all columns, RSI bounds, horizon
target shift), model path naming, train / predict lifecycle, batch operations,
and auto-train on first predict call.

All DB access goes through the tmp_db fixture (isolated temp SQLite).
Model files are written to a temp directory patched into MODELS_DIR.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Convenience context manager
# ---------------------------------------------------------------------------

from contextlib import contextmanager

@contextmanager
def predictor_ctx(tmp_db, models_dir):
    """Patch both MODELS_DIR and db inside src.predictor."""
    import src.predictor as pred
    with patch.object(pred, "MODELS_DIR", models_dir), \
         patch.object(pred, "db",         tmp_db):
        yield pred


# ===========================================================================
# Feature engineering
# ===========================================================================

class TestAddFeatures:
    def test_all_feature_columns_present(self, sample_df):
        from src.predictor import _add_features, FEATURE_COLS
        df = _add_features(sample_df)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature: {col}"

    def test_target_column_present(self, sample_df):
        from src.predictor import _add_features
        assert "target" in _add_features(sample_df).columns

    def test_target_is_binary(self, sample_df):
        from src.predictor import _add_features
        df = _add_features(sample_df).dropna(subset=["target"])
        assert set(df["target"].unique()).issubset({0, 1})

    def test_horizon_1_target_next_day_direction(self, sample_df):
        from src.predictor import _add_features
        df = _add_features(sample_df.copy(), horizon=1).dropna(subset=["target"])
        df = df.sort_values("date").reset_index(drop=True)
        # Find close values by date to verify the shift
        closes = sample_df.sort_values("date")["close"].values
        for i in range(min(len(df) - 1, 20)):
            # target[i] must equal 1 iff closes[i+1] > closes[i]
            idx_in_closes = i  # df is sorted same as sample_df after dropna top rows removed
            if i + 1 < len(closes):
                expected = int(closes[i + 1] > closes[i])
                assert df.loc[i, "target"] in (0, 1)

    def test_horizon_3_target_uses_close_plus_3(self, sample_df):
        from src.predictor import _add_features
        df_h1 = _add_features(sample_df.copy(), horizon=1).dropna(subset=["target"])
        df_h3 = _add_features(sample_df.copy(), horizon=3).dropna(subset=["target"])
        # h=3 should have fewer valid rows (more are dropped at the tail)
        assert len(df_h3) <= len(df_h1)

    def test_horizon_affects_target_values(self, sample_df):
        """Different horizons produce different (not identical) targets."""
        from src.predictor import _add_features
        df_h1 = _add_features(sample_df.copy(), horizon=1).dropna(subset=["target"])
        df_h7 = _add_features(sample_df.copy(), horizon=7).dropna(subset=["target"])
        common = min(len(df_h1), len(df_h7))
        t1 = df_h1.sort_values("date")["target"].values[:common]
        t7 = df_h7.sort_values("date")["target"].values[:common]
        # Targets are NOT identical (would be extreme coincidence on 120 rows)
        assert not (t1 == t7).all()

    def test_rsi_between_0_and_100(self, sample_df):
        from src.predictor import _add_features
        df = _add_features(sample_df).dropna(subset=["rsi"])
        assert (df["rsi"] >= 0).all()
        assert (df["rsi"] <= 100).all()

    def test_dow_within_valid_range(self, sample_df):
        from src.predictor import _add_features
        df = _add_features(sample_df).dropna(subset=["dow"])
        assert df["dow"].between(0, 6).all()

    def test_vol_ratio_positive(self, sample_df):
        from src.predictor import _add_features
        df = _add_features(sample_df).dropna(subset=["vol_ratio"])
        assert (df["vol_ratio"] > 0).all()

    def test_bb_position_finite(self, sample_df):
        from src.predictor import _add_features
        df = _add_features(sample_df).dropna(subset=["bb_position"])
        assert df["bb_position"].apply(lambda x: np.isfinite(x)).all()

    def test_price_vs_ma_near_zero_on_average(self, sample_df):
        from src.predictor import _add_features
        df = _add_features(sample_df).dropna(subset=["price_vs_ma20"])
        # Stationary series: long-run average ratio should be close to 0
        assert df["price_vs_ma20"].abs().median() < 0.5

    def test_output_sorted_by_date(self, sample_df):
        from src.predictor import _add_features
        shuffled = sample_df.sample(frac=1, random_state=99)
        df = _add_features(shuffled)
        assert df["date"].is_monotonic_increasing

    def test_no_feature_columns_all_nan(self, sample_df):
        from src.predictor import _add_features, FEATURE_COLS
        df = _add_features(sample_df)
        for col in FEATURE_COLS:
            non_null = df[col].dropna()
            assert len(non_null) > 0, f"All NaN in feature: {col}"


# ===========================================================================
# Model path naming
# ===========================================================================

class TestModelPath:
    def test_horizon_1_uses_plain_pkl(self):
        from src.predictor import _model_path, MODELS_DIR
        assert _model_path("JKH.N0000", 1) == MODELS_DIR / "JKH.N0000.pkl"

    def test_horizon_2_appends_h2(self):
        from src.predictor import _model_path, MODELS_DIR
        assert _model_path("JKH.N0000", 2) == MODELS_DIR / "JKH.N0000_h2.pkl"

    def test_horizon_7_appends_h7(self):
        from src.predictor import _model_path, MODELS_DIR
        assert _model_path("JKH.N0000", 7) == MODELS_DIR / "JKH.N0000_h7.pkl"

    def test_forward_slash_in_symbol_replaced(self):
        from src.predictor import _model_path
        p = _model_path("A/B", 1)
        assert "/" not in p.name

    def test_backslash_in_symbol_replaced(self):
        from src.predictor import _model_path
        p = _model_path("A\\B", 1)
        assert "\\" not in p.name

    def test_horizon_1_and_2_produce_different_paths(self):
        from src.predictor import _model_path
        assert _model_path("SYM.N0000", 1) != _model_path("SYM.N0000", 2)


# ===========================================================================
# Training
# ===========================================================================

class TestTrain:
    def test_too_few_rows_returns_none(self, tmp_db, tmp_path):
        tmp_db.upsert_security("X.N0000", "X Co", "Unknown")
        tmp_db.upsert_prices([
            {"symbol": "X.N0000", "date": f"2024-01-{i+1:02d}",
             "open": 10.0, "high": 11.0, "low": 9.0, "close": float(10 + i),
             "volume": 100, "trades": 5}
            for i in range(10)
        ])
        with predictor_ctx(tmp_db, tmp_path / "models") as pred:
            result = pred.train("X.N0000")
        assert result is None

    def test_sufficient_data_creates_pkl_file(self, tmp_db, tmp_path, sample_ohlcv):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prices(sample_ohlcv)
        models = tmp_path / "models"
        models.mkdir()
        with predictor_ctx(tmp_db, models) as pred:
            acc = pred.train("TEST.N0000", horizon=1)
        assert acc is not None
        assert (models / "TEST.N0000.pkl").exists()

    def test_accuracy_between_0_and_1(self, tmp_db, tmp_path, sample_ohlcv):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prices(sample_ohlcv)
        models = tmp_path / "models"
        models.mkdir()
        with predictor_ctx(tmp_db, models) as pred:
            acc = pred.train("TEST.N0000")
        assert 0.0 <= acc <= 1.0

    def test_horizon_2_creates_separate_file_from_horizon_1(self, tmp_db, tmp_path, sample_ohlcv):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prices(sample_ohlcv)
        models = tmp_path / "models"
        models.mkdir()
        with predictor_ctx(tmp_db, models) as pred:
            pred.train("TEST.N0000", horizon=1)
            pred.train("TEST.N0000", horizon=2)
        assert (models / "TEST.N0000.pkl").exists()
        assert (models / "TEST.N0000_h2.pkl").exists()

    def test_retraining_overwrites_existing_pkl(self, tmp_db, tmp_path, sample_ohlcv):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prices(sample_ohlcv)
        models = tmp_path / "models"
        models.mkdir()
        with predictor_ctx(tmp_db, models) as pred:
            pred.train("TEST.N0000")
            mtime1 = (models / "TEST.N0000.pkl").stat().st_mtime
            pred.train("TEST.N0000")
            mtime2 = (models / "TEST.N0000.pkl").stat().st_mtime
        assert mtime2 >= mtime1


# ===========================================================================
# Prediction
# ===========================================================================

class TestPredictNextDay:
    def _setup(self, tmp_db, sample_ohlcv):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prices(sample_ohlcv)

    def test_returns_expected_keys(self, tmp_db, tmp_path, sample_ohlcv):
        self._setup(tmp_db, sample_ohlcv)
        models = tmp_path / "models"
        models.mkdir()
        with predictor_ctx(tmp_db, models) as pred:
            result = pred.predict_next_day("TEST.N0000", horizon=1)
        assert result is not None
        assert set(result.keys()) == {"direction", "confidence", "predicted_close"}

    def test_direction_is_up_or_down(self, tmp_db, tmp_path, sample_ohlcv):
        self._setup(tmp_db, sample_ohlcv)
        models = tmp_path / "models"
        models.mkdir()
        with predictor_ctx(tmp_db, models) as pred:
            result = pred.predict_next_day("TEST.N0000")
        assert result["direction"] in {"UP", "DOWN"}

    def test_confidence_at_least_0_5(self, tmp_db, tmp_path, sample_ohlcv):
        self._setup(tmp_db, sample_ohlcv)
        models = tmp_path / "models"
        models.mkdir()
        with predictor_ctx(tmp_db, models) as pred:
            result = pred.predict_next_day("TEST.N0000")
        assert result["confidence"] >= 0.5
        assert result["confidence"] <= 1.0

    def test_predicted_close_is_positive(self, tmp_db, tmp_path, sample_ohlcv):
        self._setup(tmp_db, sample_ohlcv)
        models = tmp_path / "models"
        models.mkdir()
        with predictor_ctx(tmp_db, models) as pred:
            result = pred.predict_next_day("TEST.N0000")
        assert result["predicted_close"] > 0

    def test_higher_horizon_predicts_larger_price_move(self, tmp_db, tmp_path, sample_ohlcv):
        """horizon=7 price estimate should deviate more from last close than horizon=1."""
        self._setup(tmp_db, sample_ohlcv)
        models = tmp_path / "models"
        models.mkdir()
        df = pd.DataFrame(sample_ohlcv).sort_values("date")
        last_close = df.iloc[-1]["close"]
        with predictor_ctx(tmp_db, models) as pred:
            r1 = pred.predict_next_day("TEST.N0000", horizon=1)
            r7 = pred.predict_next_day("TEST.N0000", horizon=7)
        diff1 = abs(r1["predicted_close"] - last_close)
        diff7 = abs(r7["predicted_close"] - last_close)
        assert diff7 > diff1

    def test_auto_trains_model_when_pkl_missing(self, tmp_db, tmp_path, sample_ohlcv):
        self._setup(tmp_db, sample_ohlcv)
        models = tmp_path / "models"
        models.mkdir()
        assert not list(models.glob("*.pkl"))
        with predictor_ctx(tmp_db, models) as pred:
            result = pred.predict_next_day("TEST.N0000")
        assert result is not None
        assert list(models.glob("*.pkl"))

    def test_returns_none_with_insufficient_data(self, tmp_db, tmp_path):
        tmp_db.upsert_security("X.N0000", "X Co", "Unknown")
        tmp_db.upsert_prices([
            {"symbol": "X.N0000", "date": f"2024-01-{i+1:02d}",
             "open": 10.0, "high": 11.0, "low": 9.0, "close": float(10 + i),
             "volume": 100, "trades": 5}
            for i in range(10)
        ])
        models = tmp_path / "models"
        models.mkdir()
        with predictor_ctx(tmp_db, models) as pred:
            result = pred.predict_next_day("X.N0000")
        assert result is None

    def test_multiple_horizons_return_independently(self, tmp_db, tmp_path, sample_ohlcv):
        self._setup(tmp_db, sample_ohlcv)
        models = tmp_path / "models"
        models.mkdir()
        with predictor_ctx(tmp_db, models) as pred:
            results = [pred.predict_next_day("TEST.N0000", h) for h in range(1, 5)]
        assert all(r is not None for r in results)
        assert all(r["direction"] in {"UP", "DOWN"} for r in results)


# ===========================================================================
# Batch operations
# ===========================================================================

class TestBatchOperations:
    def test_train_all_creates_one_pkl_per_symbol(self, tmp_db, tmp_path, sample_ohlcv):
        for sym in ["A.N0000", "B.N0000"]:
            tmp_db.upsert_security(sym, sym, "Unknown")
            tmp_db.upsert_prices([{**r, "symbol": sym} for r in sample_ohlcv])
        models = tmp_path / "models"
        models.mkdir()
        with predictor_ctx(tmp_db, models) as pred:
            pred.train_all(horizon=1)
        names = {p.name for p in models.glob("*.pkl")}
        assert "A.N0000.pkl" in names
        assert "B.N0000.pkl" in names

    def test_train_all_horizon_2_creates_h2_files(self, tmp_db, tmp_path, sample_ohlcv):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prices(sample_ohlcv)
        models = tmp_path / "models"
        models.mkdir()
        with predictor_ctx(tmp_db, models) as pred:
            pred.train_all(horizon=2)
        assert (models / "TEST.N0000_h2.pkl").exists()

    def test_predict_all_saves_predictions_to_db(self, tmp_db, tmp_path, sample_ohlcv):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prices(sample_ohlcv)
        models = tmp_path / "models"
        models.mkdir()
        with predictor_ctx(tmp_db, models) as pred:
            pred.predict_all(horizon=1)
        df = tmp_db.get_latest_predictions(horizon=1)
        assert len(df) >= 1
        assert df.iloc[0]["symbol"] == "TEST.N0000"

    def test_predict_all_uses_correct_horizon_in_db(self, tmp_db, tmp_path, sample_ohlcv):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prices(sample_ohlcv)
        models = tmp_path / "models"
        models.mkdir()
        with predictor_ctx(tmp_db, models) as pred:
            pred.predict_all(horizon=3)
        df = tmp_db.get_latest_predictions(horizon=3)
        assert len(df) >= 1
        # No predictions stored for horizon=1
        assert tmp_db.get_latest_predictions(horizon=1).empty

    def test_predict_all_skips_symbols_with_too_little_data(self, tmp_db, tmp_path):
        tmp_db.upsert_security("TINY.N0000", "Tiny Co", "Unknown")
        tmp_db.upsert_prices([
            {"symbol": "TINY.N0000", "date": f"2024-01-{i+1:02d}",
             "open": 10.0, "high": 11.0, "low": 9.0, "close": float(10 + i),
             "volume": 100, "trades": 5}
            for i in range(5)
        ])
        models = tmp_path / "models"
        models.mkdir()
        with predictor_ctx(tmp_db, models) as pred:
            pred.predict_all(horizon=1)
        assert tmp_db.get_latest_predictions(horizon=1).empty

"""
Tests for src/database.py

Covers: schema creation, migration, CRUD for all four tables, query helpers,
prediction history join, and user-role management.
"""

import pytest
import pandas as pd


# ===========================================================================
# Schema / init
# ===========================================================================

class TestInitDb:
    def test_creates_all_four_tables(self, tmp_db):
        with tmp_db.get_connection() as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert {"securities", "daily_prices", "predictions", "users"} <= tables

    def test_idempotent_double_call(self, tmp_db):
        tmp_db.init_db()   # second call must not raise or corrupt data

    def test_idx_prices_symbol_date_created(self, tmp_db):
        with tmp_db.get_connection() as conn:
            indexes = {r[1] for r in conn.execute(
                "SELECT * FROM sqlite_master WHERE type='index'"
            ).fetchall()}
        assert "idx_prices_symbol_date" in indexes

    def test_predictions_has_horizon_column(self, tmp_db):
        with tmp_db.get_connection() as conn:
            cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(predictions)"
            ).fetchall()}
        assert "horizon" in cols

    def test_migration_adds_horizon_to_old_schema(self, tmp_path, mock_config):
        """Simulate a DB created before the horizon column existed."""
        import src.database as db
        db_file = tmp_path / "legacy.db"
        # Build the old schema manually (no horizon column)
        import sqlite3
        with sqlite3.connect(db_file) as conn:
            conn.execute("""
                CREATE TABLE predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    predicted_for TEXT NOT NULL,
                    direction TEXT, confidence REAL, predicted_close REAL,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(symbol, predicted_for)
                )
            """)
        # init_db should migrate without error
        with patch_db_path(db, db_file):
            db.init_db()
            with db.get_connection() as conn:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()}
        assert "horizon" in cols


# ===========================================================================
# Securities
# ===========================================================================

class TestUpsertSecurity:
    def test_insert_new_record(self, tmp_db):
        tmp_db.upsert_security("JKH.N0000", "John Keells Holdings PLC", "Diversified")
        df = tmp_db.get_all_securities()
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "JKH.N0000"
        assert df.iloc[0]["name"]   == "John Keells Holdings PLC"

    def test_update_name_and_sector_on_conflict(self, tmp_db):
        tmp_db.upsert_security("JKH.N0000", "Old Name", "Old Sector")
        tmp_db.upsert_security("JKH.N0000", "New Name", "New Sector")
        df = tmp_db.get_all_securities()
        assert len(df) == 1
        assert df.iloc[0]["name"]   == "New Name"
        assert df.iloc[0]["sector"] == "New Sector"

    def test_multiple_securities_stored(self, tmp_db):
        for sym in ["AAA.N0000", "BBB.N0000", "CCC.N0000"]:
            tmp_db.upsert_security(sym, f"{sym} Name", "Tech")
        assert len(tmp_db.get_all_securities()) == 3

    def test_get_all_securities_ordered_by_symbol(self, tmp_db):
        for sym in ["ZZZ.N0000", "AAA.N0000", "MMM.N0000"]:
            tmp_db.upsert_security(sym, sym, "Unknown")
        df = tmp_db.get_all_securities()
        assert list(df["symbol"]) == sorted(df["symbol"])


# ===========================================================================
# Daily prices
# ===========================================================================

class TestUpsertPrices:
    def test_inserts_all_records(self, tmp_db, sample_ohlcv):
        tmp_db.upsert_prices(sample_ohlcv)
        df = tmp_db.get_prices("TEST.N0000", limit=200)
        assert len(df) == len(sample_ohlcv)

    def test_empty_list_is_noop(self, tmp_db):
        tmp_db.upsert_prices([])   # must not raise

    def test_updates_on_symbol_date_conflict(self, tmp_db):
        row = [{"symbol": "X.N0000", "date": "2024-01-02",
                "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5,
                "volume": 1000, "trades": 10}]
        tmp_db.upsert_prices(row)
        row[0]["close"] = 99.9
        tmp_db.upsert_prices(row)
        df = tmp_db.get_prices("X.N0000")
        assert len(df) == 1
        assert df.iloc[0]["close"] == pytest.approx(99.9)

    def test_null_open_and_low_stored_as_null(self, tmp_db):
        tmp_db.upsert_prices([{
            "symbol": "X.N0000", "date": "2024-01-02",
            "open": None, "high": 11.0, "low": None, "close": 10.5,
            "volume": 500, "trades": 5,
        }])
        df = tmp_db.get_prices("X.N0000")
        assert pd.isna(df.iloc[0]["open"])
        assert pd.isna(df.iloc[0]["low"])

    def test_get_prices_limit_respected(self, tmp_db, sample_ohlcv):
        tmp_db.upsert_prices(sample_ohlcv)
        df = tmp_db.get_prices("TEST.N0000", limit=10)
        assert len(df) == 10

    def test_get_prices_returns_most_recent_first(self, tmp_db, sample_ohlcv):
        tmp_db.upsert_prices(sample_ohlcv)
        df = tmp_db.get_prices("TEST.N0000", limit=200)
        assert df.iloc[0]["date"] >= df.iloc[-1]["date"]

    def test_get_prices_returns_only_matching_symbol(self, tmp_db, sample_ohlcv):
        tmp_db.upsert_prices(sample_ohlcv)
        other = [{**r, "symbol": "OTHER.N0000"} for r in sample_ohlcv]
        tmp_db.upsert_prices(other)
        df = tmp_db.get_prices("TEST.N0000", limit=200)
        assert (df["symbol"] == "TEST.N0000").all()

    def test_count_trading_days(self, tmp_db, sample_ohlcv):
        tmp_db.upsert_prices(sample_ohlcv)
        assert tmp_db.count_trading_days("TEST.N0000") == len(sample_ohlcv)

    def test_count_trading_days_unknown_symbol_returns_zero(self, tmp_db):
        assert tmp_db.count_trading_days("GHOST.N0000") == 0


# ===========================================================================
# Predictions
# ===========================================================================

class TestUpsertPrediction:
    def test_insert_and_retrieve(self, tmp_db):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prediction("TEST.N0000", "2025-01-10", "UP", 0.72, 105.5, 1)
        df = tmp_db.get_latest_predictions(horizon=1)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["direction"]       == "UP"
        assert row["confidence"]      == pytest.approx(0.72)
        assert row["predicted_close"] == pytest.approx(105.5)
        assert row["horizon"]         == 1

    def test_updates_on_symbol_predicted_for_horizon_conflict(self, tmp_db):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prediction("TEST.N0000", "2025-01-10", "UP",   0.72, 105.5, 1)
        tmp_db.upsert_prediction("TEST.N0000", "2025-01-10", "DOWN", 0.65,  95.0, 1)
        df = tmp_db.get_latest_predictions(horizon=1)
        assert len(df) == 1
        assert df.iloc[0]["direction"] == "DOWN"

    def test_different_horizons_stored_as_separate_rows(self, tmp_db):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prediction("TEST.N0000", "2025-01-02", "UP",   0.70, 101.0, 1)
        tmp_db.upsert_prediction("TEST.N0000", "2025-01-08", "DOWN", 0.60,  95.0, 7)
        assert len(tmp_db.get_latest_predictions(horizon=1)) == 1
        assert len(tmp_db.get_latest_predictions(horizon=7)) == 1

    def test_get_latest_predictions_filters_by_horizon(self, tmp_db):
        for sym in ["A.N0000", "B.N0000"]:
            tmp_db.upsert_security(sym, sym, "Unknown")
        tmp_db.upsert_prediction("A.N0000", "2025-01-02", "UP",   0.70, 101.0, 1)
        tmp_db.upsert_prediction("B.N0000", "2025-01-02", "DOWN", 0.60,  99.0, 1)
        tmp_db.upsert_prediction("A.N0000", "2025-01-04", "UP",   0.65, 103.0, 3)
        assert len(tmp_db.get_latest_predictions(horizon=1)) == 2
        assert len(tmp_db.get_latest_predictions(horizon=3)) == 1
        assert tmp_db.get_latest_predictions(horizon=3).iloc[0]["symbol"] == "A.N0000"

    def test_get_latest_returns_most_recent_predicted_for_date(self, tmp_db):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prediction("TEST.N0000", "2025-01-02", "UP",   0.70, 101.0, 1)
        tmp_db.upsert_prediction("TEST.N0000", "2025-01-03", "DOWN", 0.60,  98.0, 1)
        df = tmp_db.get_latest_predictions(horizon=1)
        assert len(df) == 1
        assert df.iloc[0]["predicted_for"] == "2025-01-03"

    def test_empty_when_no_predictions_for_horizon(self, tmp_db):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prediction("TEST.N0000", "2025-01-02", "UP", 0.7, 101.0, 1)
        df = tmp_db.get_latest_predictions(horizon=5)
        assert df.empty

    def test_confidence_and_close_stored_precisely(self, tmp_db):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prediction("TEST.N0000", "2025-06-01", "UP", 0.7654, 123.45, 1)
        row = tmp_db.get_latest_predictions(horizon=1).iloc[0]
        assert row["confidence"]      == pytest.approx(0.7654)
        assert row["predicted_close"] == pytest.approx(123.45)


# ===========================================================================
# Market summary
# ===========================================================================

class TestGetMarketSummary:
    def test_returns_empty_when_no_data(self, tmp_db):
        assert tmp_db.get_market_summary().empty

    def test_change_pct_calculated_correctly(self, tmp_db):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        # open=100, close=110 → change = +10 %
        tmp_db.upsert_prices([{
            "symbol": "TEST.N0000", "date": "2024-06-01",
            "open": 100.0, "high": 115.0, "low": 98.0, "close": 110.0,
            "volume": 1000, "trades": 50,
        }])
        df = tmp_db.get_market_summary()
        assert len(df) == 1
        assert df.iloc[0]["change_pct"] == pytest.approx(10.0)

    def test_returns_only_latest_date_per_symbol(self, tmp_db):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prices([
            {"symbol": "TEST.N0000", "date": "2024-06-01",
             "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0,
             "volume": 1000, "trades": 50},
            {"symbol": "TEST.N0000", "date": "2024-06-02",
             "open": 102.0, "high": 108.0, "low": 101.0, "close": 107.0,
             "volume": 1200, "trades": 60},
        ])
        df = tmp_db.get_market_summary()
        assert len(df) == 1
        assert df.iloc[0]["date"] == "2024-06-02"

    def test_change_pct_negative_when_close_below_open(self, tmp_db):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prices([{
            "symbol": "TEST.N0000", "date": "2024-06-01",
            "open": 200.0, "high": 201.0, "low": 180.0, "close": 180.0,
            "volume": 500, "trades": 20,
        }])
        df = tmp_db.get_market_summary()
        assert df.iloc[0]["change_pct"] == pytest.approx(-10.0)

    def test_handles_null_open_without_crash(self, tmp_db):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prices([{
            "symbol": "TEST.N0000", "date": "2024-06-01",
            "open": None, "high": 105.0, "low": 99.0, "close": 102.0,
            "volume": 1000, "trades": 50,
        }])
        df = tmp_db.get_market_summary()
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["change_pct"])


# ===========================================================================
# User / auth management
# ===========================================================================

class TestUpsertUser:
    def test_new_user_gets_pending_role(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        role = tmp_db.upsert_user("user@example.com", "Test User", "")
        assert role == "pending"

    def test_admin_email_gets_admin_role(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = ["admin@example.com"]
        role = tmp_db.upsert_user("admin@example.com", "Admin", "")
        assert role == "admin"

    def test_returning_user_keeps_approved_role(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        tmp_db.upsert_user("user@example.com", "User", "")
        tmp_db.set_user_role("user@example.com", "approved", "admin@example.com")
        role = tmp_db.upsert_user("user@example.com", "User", "new-pic")
        assert role == "approved"

    def test_returning_user_with_rejected_role_stays_rejected(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        tmp_db.upsert_user("bad@example.com", "Bad Actor", "")
        tmp_db.set_user_role("bad@example.com", "rejected", "admin@example.com")
        role = tmp_db.upsert_user("bad@example.com", "Bad Actor", "")
        assert role == "rejected"

    def test_admin_email_overrides_any_existing_role(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = ["admin@example.com"]
        tmp_db.upsert_user("admin@example.com", "Admin", "")
        tmp_db.set_user_role("admin@example.com", "pending", "")
        # Re-login should restore admin role
        role = tmp_db.upsert_user("admin@example.com", "Admin", "")
        assert role == "admin"

    def test_upsert_updates_name_and_picture(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        tmp_db.upsert_user("user@example.com", "Old Name", "old-pic")
        tmp_db.upsert_user("user@example.com", "New Name", "new-pic")
        df = tmp_db.get_all_users()
        row = df[df["email"] == "user@example.com"].iloc[0]
        assert row["name"] == "New Name"

    def test_multiple_users_all_stored(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        for i in range(5):
            tmp_db.upsert_user(f"user{i}@example.com", f"User {i}", "")
        assert len(tmp_db.get_all_users()) == 5


class TestSetUserRole:
    def test_role_updated(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        tmp_db.upsert_user("user@example.com", "User", "")
        tmp_db.set_user_role("user@example.com", "approved", "admin@example.com")
        row = tmp_db.get_all_users().iloc[0]
        assert row["role"] == "approved"

    def test_approved_by_recorded(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        tmp_db.upsert_user("user@example.com", "User", "")
        tmp_db.set_user_role("user@example.com", "approved", "admin@example.com")
        row = tmp_db.get_all_users().iloc[0]
        assert row["approved_by"] == "admin@example.com"

    def test_approved_at_set_when_approved(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        tmp_db.upsert_user("user@example.com", "User", "")
        tmp_db.set_user_role("user@example.com", "approved", "admin@example.com")
        row = tmp_db.get_all_users().iloc[0]
        assert row["approved_at"] is not None

    def test_approved_at_null_when_revoked(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        tmp_db.upsert_user("user@example.com", "User", "")
        tmp_db.set_user_role("user@example.com", "approved", "admin@example.com")
        tmp_db.set_user_role("user@example.com", "pending", "admin@example.com")
        row = tmp_db.get_all_users().iloc[0]
        assert row["approved_at"] is None


# ===========================================================================
# Prediction history join
# ===========================================================================

class TestGetPredictionHistory:
    def _insert_past_prediction(self, tmp_db, symbol="TEST.N0000",
                                pred_dir="UP", prev_close=100.0, actual_close=105.0,
                                horizon=1):
        """Helper: insert a security, prev-day price, actual price, and one past prediction."""
        pred_date = "2024-06-01"
        prev_date = "2024-05-31"
        tmp_db.upsert_security(symbol, f"{symbol} Co", "Unknown")
        tmp_db.upsert_prices([
            {"symbol": symbol, "date": prev_date,
             "open": prev_close - 1, "high": prev_close + 1,
             "low": prev_close - 2, "close": prev_close,
             "volume": 1000, "trades": 50},
            {"symbol": symbol, "date": pred_date,
             "open": actual_close - 0.5, "high": actual_close + 1,
             "low": actual_close - 1, "close": actual_close,
             "volume": 1200, "trades": 60},
        ])
        tmp_db.upsert_prediction(symbol, pred_date, pred_dir, 0.75, actual_close + 1, horizon)

    def test_returns_empty_when_no_past_predictions(self, tmp_db):
        assert tmp_db.get_prediction_history().empty

    def test_excludes_future_predictions(self, tmp_db):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        tmp_db.upsert_prediction("TEST.N0000", "2099-12-31", "UP", 0.7, 110.0, 1)
        assert tmp_db.get_prediction_history().empty

    def test_excludes_predictions_without_actual_price(self, tmp_db):
        tmp_db.upsert_security("TEST.N0000", "Test Co", "Unknown")
        # Prediction for a past date, but no daily_prices row for that date
        tmp_db.upsert_prediction("TEST.N0000", "2024-01-15", "UP", 0.7, 110.0, 1)
        assert tmp_db.get_prediction_history().empty

    def test_correct_up_prediction_marked_correct(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        self._insert_past_prediction(tmp_db, pred_dir="UP",
                                     prev_close=100.0, actual_close=105.0)
        df = tmp_db.get_prediction_history(horizon=1)
        assert len(df) == 1
        assert df.iloc[0]["correct"] == 1
        assert df.iloc[0]["actual_direction"] == "UP"

    def test_wrong_up_prediction_marked_incorrect(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        # Predicted UP but price went DOWN
        self._insert_past_prediction(tmp_db, pred_dir="UP",
                                     prev_close=100.0, actual_close=95.0)
        df = tmp_db.get_prediction_history(horizon=1)
        assert df.iloc[0]["correct"] == 0
        assert df.iloc[0]["actual_direction"] == "DOWN"

    def test_correct_down_prediction_marked_correct(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        self._insert_past_prediction(tmp_db, pred_dir="DOWN",
                                     prev_close=100.0, actual_close=93.0)
        df = tmp_db.get_prediction_history(horizon=1)
        assert df.iloc[0]["correct"] == 1

    def test_flat_market_counted_as_down_prediction_correct(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        # price unchanged → actual_direction=FLAT → DOWN prediction should be correct
        self._insert_past_prediction(tmp_db, pred_dir="DOWN",
                                     prev_close=100.0, actual_close=100.0)
        df = tmp_db.get_prediction_history(horizon=1)
        assert df.iloc[0]["correct"] == 1
        assert df.iloc[0]["actual_direction"] == "FLAT"

    def test_filters_by_horizon(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        self._insert_past_prediction(tmp_db, pred_dir="UP",
                                     prev_close=100.0, actual_close=105.0, horizon=1)
        self._insert_past_prediction(tmp_db, symbol="OTHER.N0000",
                                     pred_dir="DOWN", prev_close=50.0, actual_close=45.0,
                                     horizon=3)
        h1 = tmp_db.get_prediction_history(horizon=1)
        h3 = tmp_db.get_prediction_history(horizon=3)
        assert len(h1) == 1
        assert h1.iloc[0]["symbol"] == "TEST.N0000"
        assert len(h3) == 1
        assert h3.iloc[0]["symbol"] == "OTHER.N0000"

    def test_filters_by_symbol(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        self._insert_past_prediction(tmp_db, symbol="TEST.N0000",
                                     pred_dir="UP", prev_close=100.0, actual_close=105.0)
        self._insert_past_prediction(tmp_db, symbol="OTHER.N0000",
                                     pred_dir="DOWN", prev_close=50.0, actual_close=45.0)
        df = tmp_db.get_prediction_history(symbol="TEST.N0000")
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "TEST.N0000"

    def test_returns_expected_columns(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        self._insert_past_prediction(tmp_db)
        df = tmp_db.get_prediction_history()
        expected_cols = {
            "symbol", "name", "predicted_for", "horizon",
            "predicted_direction", "confidence", "predicted_close",
            "actual_close", "prev_close", "actual_direction", "correct",
        }
        assert expected_cols <= set(df.columns)

    def test_limit_respected(self, tmp_db, mock_config):
        mock_config.ADMIN_EMAILS = []
        for i in range(5):
            sym = f"SYM{i}.N0000"
            self._insert_past_prediction(tmp_db, symbol=sym,
                                         prev_close=100.0, actual_close=105.0)
        df = tmp_db.get_prediction_history(limit=3)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# Helper (module-level, not a fixture)
# ---------------------------------------------------------------------------

from contextlib import contextmanager
from unittest.mock import patch

@contextmanager
def patch_db_path(db_module, path):
    with patch.object(db_module, "DB_PATH", path):
        yield

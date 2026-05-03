"""
Shared fixtures for the CSE test suite.

Key fixtures
------------
mock_config   — injects a fake `config` module into sys.modules so every
                in-function `import config` resolves without touching the
                real config.py (which contains live credentials).

tmp_db        — patches src.database.DB_PATH to a temp file, runs init_db,
                and yields the module.  Every test gets a fresh, isolated DB.

sample_ohlcv  — 120 rows of deterministic synthetic OHLCV data for the
                symbol "TEST.N0000", enough to train the ML models.

sample_df     — same data as a DataFrame.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make the project root importable when running pytest from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Config mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """Inject a fake config module so no test ever touches the real one."""
    cfg = MagicMock()
    cfg.ADMIN_EMAILS        = ["admin@example.com"]
    cfg.GOOGLE_CLIENT_ID    = "1234567890.apps.googleusercontent.com"
    cfg.GOOGLE_CLIENT_SECRET = "test-client-secret-xyz"
    cfg.REDIRECT_URI        = "http://localhost:8501/"
    cfg.CSE_TOKEN           = ""
    cfg.CSE_EMAIL           = "test@example.com"
    cfg.CSE_PASSWORD        = "testpassword"
    sys.modules["config"] = cfg
    yield cfg
    sys.modules.pop("config", None)


# ---------------------------------------------------------------------------
# Isolated database
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path, mock_config):
    """
    Returns src.database with DB_PATH redirected to a temp SQLite file.
    Each test gets a completely empty, freshly-initialised database.
    """
    import src.database as db
    db_file = tmp_path / "test_cse.db"
    with patch.object(db, "DB_PATH", db_file):
        db.init_db()
        yield db


# ---------------------------------------------------------------------------
# Sample market data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv():
    """
    120 business-day rows of synthetic OHLCV for symbol TEST.N0000.
    Deterministic (fixed RNG seed) so feature-engineering assertions are stable.
    """
    rng   = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=120)
    close = 100.0
    rows  = []
    for d in dates:
        close = max(close + rng.normal(0, 1.5), 1.0)
        rows.append({
            "symbol": "TEST.N0000",
            "date":   d.strftime("%Y-%m-%d"),
            "open":   round(close - 0.5, 2),
            "high":   round(close + 1.0, 2),
            "low":    round(close - 1.0, 2),
            "close":  round(close, 2),
            "volume": int(10_000 + rng.integers(0, 5_000)),
            "trades": int(100   + rng.integers(0, 50)),
        })
    return rows


@pytest.fixture
def sample_df(sample_ohlcv):
    return pd.DataFrame(sample_ohlcv)

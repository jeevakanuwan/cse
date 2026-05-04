import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "cse.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                email        TEXT PRIMARY KEY,
                name         TEXT,
                picture      TEXT,
                role         TEXT NOT NULL DEFAULT 'pending',
                created_at   TEXT DEFAULT (datetime('now')),
                approved_at  TEXT,
                approved_by  TEXT
            );

            CREATE TABLE IF NOT EXISTS securities (
                symbol      TEXT PRIMARY KEY,
                name        TEXT,
                sector      TEXT,
                updated_at  TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_prices (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                date        TEXT NOT NULL,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                volume      INTEGER,
                trades      INTEGER,
                UNIQUE(symbol, date)
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT NOT NULL,
                predicted_for   TEXT NOT NULL,
                horizon         INT  NOT NULL DEFAULT 1,
                direction       TEXT,
                confidence      REAL,
                predicted_close REAL,
                created_at      TEXT DEFAULT (datetime('now')),
                UNIQUE(symbol, predicted_for, horizon)
            );

            CREATE INDEX IF NOT EXISTS idx_prices_symbol_date
                ON daily_prices(symbol, date);
        """)
        # Migration: add horizon column to existing DBs that predate this change
        try:
            conn.execute("ALTER TABLE predictions ADD COLUMN horizon INT NOT NULL DEFAULT 1")
            conn.commit()
        except Exception:
            pass

        # Migration: rebuild predictions table so UNIQUE constraint includes horizon
        try:
            import re as _re
            cur = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='predictions'"
            )
            row = cur.fetchone()
            if row:
                m = _re.search(r'UNIQUE\s*\(([^)]+)\)', row[0], _re.IGNORECASE)
                if m and 'horizon' not in m.group(1):
                    conn.executescript("""
                        CREATE TABLE predictions_new (
                            id              INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol          TEXT NOT NULL,
                            predicted_for   TEXT NOT NULL,
                            horizon         INT  NOT NULL DEFAULT 1,
                            direction       TEXT,
                            confidence      REAL,
                            predicted_close REAL,
                            created_at      TEXT DEFAULT (datetime('now')),
                            UNIQUE(symbol, predicted_for, horizon)
                        );
                        INSERT OR IGNORE INTO predictions_new
                            (id, symbol, predicted_for, horizon, direction, confidence, predicted_close, created_at)
                        SELECT id, symbol, predicted_for, COALESCE(horizon, 1),
                               direction, confidence, predicted_close, created_at
                        FROM predictions;
                        DROP TABLE predictions;
                        ALTER TABLE predictions_new RENAME TO predictions;
                    """)
        except Exception:
            pass


def upsert_security(symbol: str, name: str, sector: str):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO securities(symbol, name, sector, updated_at)
            VALUES(?, ?, ?, datetime('now'))
            ON CONFLICT(symbol) DO UPDATE SET
                name       = excluded.name,
                sector     = excluded.sector,
                updated_at = excluded.updated_at
        """, (symbol, name, sector))


def upsert_prices(records: list[dict]):
    if not records:
        return
    with get_connection() as conn:
        conn.executemany("""
            INSERT INTO daily_prices(symbol, date, open, high, low, close, volume, trades)
            VALUES(:symbol, :date, :open, :high, :low, :close, :volume, :trades)
            ON CONFLICT(symbol, date) DO UPDATE SET
                open   = excluded.open,
                high   = excluded.high,
                low    = excluded.low,
                close  = excluded.close,
                volume = excluded.volume,
                trades = excluded.trades
        """, records)


def upsert_prediction(symbol: str, predicted_for: str, direction: str,
                      confidence: float, predicted_close: float, horizon: int = 1):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO predictions(symbol, predicted_for, horizon, direction, confidence, predicted_close)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, predicted_for, horizon) DO UPDATE SET
                direction       = excluded.direction,
                confidence      = excluded.confidence,
                predicted_close = excluded.predicted_close,
                created_at      = datetime('now')
        """, (symbol, predicted_for, horizon, direction, confidence, predicted_close))


def get_all_securities() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM securities ORDER BY symbol", conn)


def get_prices(symbol: str, limit: int = 365) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT * FROM daily_prices
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT ?
        """, conn, params=(symbol, limit))


def get_latest_predictions(horizon: int = 1) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT p.*, s.name, s.sector
            FROM predictions p
            JOIN securities s ON s.symbol = p.symbol
            WHERE p.horizon = ?
              AND p.predicted_for = (
                SELECT MAX(predicted_for) FROM predictions WHERE horizon = ?
            )
            ORDER BY p.confidence DESC
        """, conn, params=(horizon, horizon))


def get_prediction_history(symbol: str = None, horizon: int = None,
                           limit: int = 10000) -> pd.DataFrame:
    """
    Past predictions joined with actual outcomes.
    Only includes rows where the predicted date has passed AND actual price
    data for that date exists in daily_prices.

    Returns columns:
      symbol, name, predicted_for, horizon, predicted_direction, confidence,
      predicted_close, actual_close, prev_close, actual_direction, correct (0/1)
    """
    with get_connection() as conn:
        filters, params = [], []
        if symbol:
            filters.append("p.symbol = ?");  params.append(symbol)
        if horizon is not None:
            filters.append("p.horizon = ?"); params.append(horizon)
        extra = ("AND " + " AND ".join(filters)) if filters else ""
        params.append(limit)
        return pd.read_sql(f"""
            SELECT
                p.symbol,
                s.name,
                p.predicted_for,
                p.horizon,
                p.direction            AS predicted_direction,
                p.confidence,
                p.predicted_close,
                actual.close           AS actual_close,
                prev_day.close         AS prev_close,
                CASE
                    WHEN actual.close >  prev_day.close THEN 'UP'
                    WHEN actual.close <  prev_day.close THEN 'DOWN'
                    ELSE 'FLAT'
                END                    AS actual_direction,
                CASE
                    WHEN p.direction = 'UP'   AND actual.close >  prev_day.close THEN 1
                    WHEN p.direction = 'DOWN' AND actual.close <= prev_day.close THEN 1
                    ELSE 0
                END                    AS correct
            FROM predictions p
            JOIN  securities s     ON s.symbol    = p.symbol
            LEFT JOIN daily_prices actual
                ON actual.symbol = p.symbol
               AND actual.date   = p.predicted_for
            LEFT JOIN daily_prices prev_day
                ON prev_day.symbol = p.symbol
               AND prev_day.date   = (
                    SELECT MAX(date) FROM daily_prices
                    WHERE  symbol = p.symbol AND date < p.predicted_for
               )
            WHERE p.predicted_for < date('now')
              AND actual.close    IS NOT NULL
              AND prev_day.close  IS NOT NULL
              {extra}
            ORDER BY p.predicted_for DESC
            LIMIT ?
        """, conn, params=tuple(params))


def get_market_summary() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT
                dp.symbol,
                s.name,
                s.sector,
                dp.close,
                dp.open,
                dp.high,
                dp.low,
                dp.volume,
                dp.date,
                ROUND((dp.close - dp.open) / NULLIF(dp.open, 0) * 100, 2) AS change_pct
            FROM daily_prices dp
            JOIN securities s ON s.symbol = dp.symbol
            WHERE dp.date = (SELECT MAX(date) FROM daily_prices WHERE symbol = dp.symbol)
            ORDER BY ABS(change_pct) DESC
        """, conn)


def count_trading_days(symbol: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM daily_prices WHERE symbol = ?", (symbol,)
        )
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# User / auth management
# ---------------------------------------------------------------------------

def upsert_user(email: str, name: str, picture: str) -> str:
    """
    Insert or update a user on login.
    - Emails listed in config.ADMIN_EMAILS are always assigned role='admin'.
    - Returning users keep their existing role.
    - New users get role='pending'.
    Returns the user's current role.
    """
    try:
        from config import ADMIN_EMAILS
    except ImportError:
        ADMIN_EMAILS = []

    with get_connection() as conn:
        row = conn.execute(
            "SELECT role FROM users WHERE email = ?", (email,)
        ).fetchone()

        if email in ADMIN_EMAILS:
            role = "admin"
        elif row:
            role = row[0]
        else:
            role = "pending"

        conn.execute("""
            INSERT INTO users(email, name, picture, role)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                name    = excluded.name,
                picture = excluded.picture,
                role    = ?
        """, (email, name, picture, role, role))

    return role


def get_all_users() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(
            "SELECT email, name, role, created_at, approved_at, approved_by "
            "FROM users ORDER BY created_at DESC",
            conn,
        )


def set_user_role(email: str, role: str, approved_by: str = ""):
    approved_at = "datetime('now')" if role == "approved" else "NULL"
    with get_connection() as conn:
        conn.execute(f"""
            UPDATE users
            SET role        = ?,
                approved_by = ?,
                approved_at = {approved_at}
            WHERE email = ?
        """, (role, approved_by, email))

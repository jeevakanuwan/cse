"""
Fetches market data from the Colombo Stock Exchange (cse.lk).

Confirmed endpoints (from browser network capture):
  GET  /api/allSecurityCode          — list of all 319 securities
  POST /api/allSectors               — sector list  body: {"period":"4"}
  POST /api/charts (form-encoded)    — OHLCV history (requires login)
       body: symbol=X&fromDate=DD-MM-YYYY&toDate=DD-MM-YYYY&period=1

Authentication: POST /api/auth/signin  body: {"email":..,"password":..}
The server sets an accessToken cookie used by subsequent requests.
"""

import time
import logging
from datetime import datetime, timedelta

import requests

from . import database as db

logger = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({
    "accept":           "application/json, text/plain, */*",
    "accept-language":  "en",
    "content-type":     "application/json",
    "origin":           "https://www.cse.lk",
    "referer":          "https://www.cse.lk/",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0"
    ),
})

BASE          = "https://www.cse.lk/api"
REQUEST_DELAY = 1.2
_logged_in    = False


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def login(email: str = None, password: str = None) -> bool:
    """
    Authenticate with cse.lk.

    Priority:
    1. CSE_TOKEN in config.py  (copy from browser DevTools → Cookies)
    2. Credential-based login via identity.cse.lk (ROPC grant)

    CSE uses an external identity provider, so token-based auth is the
    most reliable path.  The token expires every ~3 hours.
    """
    global _logged_in

    try:
        import config as cfg
    except ImportError:
        logger.error("config.py not found.")
        return False

    # --- Option 1: direct token from browser ---
    raw_token = getattr(cfg, "CSE_TOKEN", "").strip()
    if raw_token:
        # user may have pasted the full cookie string — extract just the JWT
        token = raw_token.split(";")[0].strip()
        SESSION.cookies.set("accessToken", token, domain="www.cse.lk")
        SESSION.headers.update({"authorization": f"Bearer {token}"})
        logger.info("Using accessToken from config.py.")
        _logged_in = True
        return True

    # --- Option 2: ROPC grant via identity server ---
    if email is None:
        email    = getattr(cfg, "CSE_EMAIL",    "")
        password = getattr(cfg, "CSE_PASSWORD", "")

    CLIENT_ID = "254dfe77-698a-4e32-82d6-5caa6e6dd5d5"
    try:
        resp = requests.post(
            "https://identity.cse.lk/connect/token",
            data={
                "grant_type": "password",
                "username":   email,
                "password":   password,
                "client_id":  CLIENT_ID,
                "scope":      "openid email EFutures CSE",
            },
            headers={"content-type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        if resp.status_code == 200:
            token = resp.json().get("access_token", "")
            if token:
                SESSION.cookies.set("accessToken", token, domain="www.cse.lk")
                SESSION.headers.update({"authorization": f"Bearer {token}"})
                logger.info("Logged in via identity.cse.lk (ROPC).")
                _logged_in = True
                return True
    except Exception as exc:
        logger.debug("ROPC login failed: %s", exc)

    logger.error(
        "Login failed.\n"
        "  → Open cse.lk in browser, press F12 → Application → Cookies → "
        "copy 'accessToken' value → paste into config.py as CSE_TOKEN."
    )
    return False


def ensure_logged_in():
    """Call before any authenticated request."""
    if not _logged_in:
        login()


# ---------------------------------------------------------------------------
# Low-level request helpers
# ---------------------------------------------------------------------------

def _post(path: str, body: dict, retries: int = 3) -> dict | list | None:
    """POST with JSON body."""
    url = f"{BASE}{path}"
    for attempt in range(retries):
        try:
            resp = SESSION.post(url, json=body, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP %s — POST %s (attempt %d)", exc.response.status_code, path, attempt + 1)
        except Exception as exc:
            logger.warning("POST %s failed: %s (attempt %d)", path, exc, attempt + 1)
        time.sleep(2 ** attempt)
    return None


def _post_form(path: str, data: dict, retries: int = 3) -> dict | list | None:
    """POST with application/x-www-form-urlencoded body (required by /api/charts)."""
    url = f"{BASE}{path}"
    headers = {"content-type": "application/x-www-form-urlencoded",
               "accept": "application/json"}
    for attempt in range(retries):
        try:
            resp = SESSION.post(url, data=data, headers=headers, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP %s — POST(form) %s (attempt %d)", exc.response.status_code, path, attempt + 1)
        except Exception as exc:
            logger.warning("POST(form) %s failed: %s (attempt %d)", path, exc, attempt + 1)
        time.sleep(2 ** attempt)
    return None


def _get(path: str, params: dict = None, retries: int = 3) -> dict | list | None:
    """GET with optional query params."""
    url = f"{BASE}{path}"
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP %s — GET %s (attempt %d)", exc.response.status_code, path, attempt + 1)
        except Exception as exc:
            logger.warning("GET %s failed: %s (attempt %d)", path, exc, attempt + 1)
        time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Sectors
# ---------------------------------------------------------------------------

def fetch_sectors() -> list[dict]:
    """POST /api/allSectors  body: {"period":"4"}"""
    data = _post("/allSectors", {"period": "4"})
    if not data:
        return []
    return data if isinstance(data, list) else data.get("data", [])


# ---------------------------------------------------------------------------
# Securities list
# ---------------------------------------------------------------------------

def fetch_securities() -> list[dict]:
    """GET /api/allSecurityCode enriched with sector info from /api/allSectors."""
    data = _get("/allSecurityCode")
    if not data:
        logger.error("Could not fetch securities.")
        return []

    sector_map = _build_sector_map()
    items = data if isinstance(data, list) else data.get("data", [])

    securities = []
    for item in items:
        symbol = (
            item.get("symbol") or item.get("stockCode") or
            item.get("securityCode") or item.get("code") or item.get("id", "")
        )
        name = (
            item.get("name") or item.get("companyName") or
            item.get("securityName") or symbol
        )
        sector = sector_map.get(symbol) or "Unknown"
        if symbol:
            securities.append({"symbol": symbol, "name": name, "sector": sector})

    logger.info("Fetched %d securities.", len(securities))
    return securities


def _build_sector_map() -> dict:
    raw = _post("/allSectors", {"period": "4"})
    if not raw:
        return {}

    items = raw if isinstance(raw, list) else raw.get("data", [])
    mapping = {}
    for sector in items:
        sector_name = (
            sector.get("sector") or sector.get("sectorName") or
            sector.get("name") or "Unknown"
        )
        stocks = (
            sector.get("securities") or sector.get("stocks") or
            sector.get("instruments") or sector.get("companies") or []
        )
        for stock in stocks:
            sym = (
                stock.get("symbol") or stock.get("stockCode") or
                stock.get("securityCode") or stock.get("code") or ""
            )
            if sym:
                mapping[sym] = sector_name

        # flat-list format: each row has both symbol + sector
        sym = (
            sector.get("symbol") or sector.get("stockCode") or
            sector.get("securityCode") or ""
        )
        if sym and sector_name != "Unknown":
            mapping[sym] = sector_name

    logger.info("Sector map built — %d symbols mapped.", len(mapping))
    return mapping


# ---------------------------------------------------------------------------
# Historical OHLCV  — POST /api/charts (form-encoded, requires login)
# ---------------------------------------------------------------------------

def fetch_history(symbol: str, from_date: str, to_date: str) -> list[dict]:
    """
    POST /api/charts with form data.
    from_date / to_date: YYYY-MM-DD  (converted internally to DD-MM-YYYY).
    Requires a valid accessToken cookie — call login() first.

    Response fields confirmed from CSE API:
      open, high, low, close   — floats (may be null)
      shareVolume              — int
      tradeVolume              — number of trades
      tradeDate                — millisecond Unix timestamp
    """
    ensure_logged_in()

    data = _post_form("/charts", {
        "symbol":   symbol,
        "fromDate": _to_cse_date(from_date),
        "toDate":   _to_cse_date(to_date),
        "period":   "1",
    })

    if not data:
        return []

    items = data if isinstance(data, list) else data.get("data", [])
    if not items:
        return []

    records = []
    for item in items:
        # tradeDate is milliseconds since epoch
        ts_ms = item.get("tradeDate")
        if not ts_ms:
            continue
        date = datetime.utcfromtimestamp(int(ts_ms) / 1000).strftime("%Y-%m-%d")

        records.append({
            "symbol": symbol,
            "date":   date,
            "open":   item.get("open"),
            "high":   item.get("high"),
            "low":    item.get("low"),
            "close":  item.get("close"),
            "volume": item.get("shareVolume") or 0,
            "trades": item.get("tradeVolume") or 0,
        })
    return records


# ---------------------------------------------------------------------------
# Daily trade summary
# ---------------------------------------------------------------------------

def fetch_daily_summary(date: str = None) -> list[dict]:
    """POST /api/tradeSummary  body: {"date": "YYYY-MM-DD"}"""
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")

    data = None
    for endpoint in ["/tradeSummary", "/tradeToTrade"]:
        data = _post(endpoint, {"date": date})
        if data:
            break

    if not data:
        return []

    items = data if isinstance(data, list) else data.get("data", [])
    records = []
    for item in items:
        symbol = item.get("symbol") or item.get("stockCode", "")
        if not symbol:
            continue
        records.append({
            "symbol": symbol,
            "date":   date,
            "open":   _safe_float(item, "openingPrice"),
            "high":   _safe_float(item, "highPrice"),
            "low":    _safe_float(item, "lowPrice"),
            "close":  (
                _safe_float(item, "lastTradedPrice") or
                _safe_float(item, "closingPrice")
            ),
            "volume": _safe_int(item, "volume"),
            "trades": _safe_int(item, "noOfTrades"),
        })
    return records


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def bootstrap_all_securities(years: int = 3):
    """First-time setup: download all securities + their full price history."""
    db.init_db()

    if not login():
        logger.error("Cannot bootstrap without a valid CSE login. Update config.py.")
        return

    securities = fetch_securities()
    if not securities:
        return

    for sec in securities:
        db.upsert_security(sec["symbol"], sec["name"], sec["sector"])

    to_date   = datetime.today().strftime("%Y-%m-%d")
    from_date = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    total     = len(securities)

    for idx, sec in enumerate(securities, 1):
        symbol = sec["symbol"]
        logger.info("[%d/%d] %s …", idx, total, symbol)
        records = fetch_history(symbol, from_date, to_date)
        if records:
            db.upsert_prices(records)
        time.sleep(REQUEST_DELAY)

    logger.info("Bootstrap complete — %d securities.", total)


def refresh_today():
    """Lightweight daily refresh — fetch today's trade summary for all stocks."""
    db.init_db()
    today = datetime.today().strftime("%Y-%m-%d")
    logger.info("Refreshing daily summary for %s …", today)
    records = fetch_daily_summary(today)
    if records:
        db.upsert_prices(records)
        logger.info("Saved %d records for %s.", len(records), today)
    else:
        logger.warning("No data returned for %s — market may be closed.", today)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_cse_date(date_str: str) -> str:
    """YYYY-MM-DD  →  DD-MM-YYYY  (CSE's required format)."""
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%d-%m-%Y")


def _parse_cse_date(raw: str) -> str | None:
    """Try multiple date formats CSE might return; output YYYY-MM-DD."""
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d %b %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            continue
    return None


def _safe_float(d: dict, key: str) -> float | None:
    try:
        v = d.get(key)
        return float(v) if v not in (None, "", "N/A") else None
    except (ValueError, TypeError):
        return None


def _safe_int(d: dict, key: str) -> int:
    try:
        v = d.get(key)
        return int(float(v)) if v not in (None, "", "N/A") else 0
    except (ValueError, TypeError):
        return 0

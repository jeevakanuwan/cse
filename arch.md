# CSE Stock Prediction — Architecture Document

## 1. System Overview

A full-stack web application that scrapes historical trading data from the Colombo Stock Exchange (cse.lk), trains a per-stock machine-learning model for each of the ~319 listed securities, and surfaces next-day price direction predictions through a role-gated Streamlit dashboard.

The system runs entirely on a single AWS EC2 t2.micro instance (Ubuntu 22.04) in the ap-southeast-1 (Singapore) region, chosen for lowest latency to Sri Lanka. There are no microservices, no message queues, and no external databases — the entire persistence layer is a single SQLite file on the instance's EBS root volume.

---

## 2. High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     AWS EC2 t2.micro                        │
│                    ap-southeast-1                           │
│                                                             │
│  ┌──────────┐   port 80   ┌──────────────────────────────┐ │
│  │  nginx   │────────────▶│  Streamlit (port 8501)       │ │
│  │ (reverse │  WebSocket  │  app.py                      │ │
│  │  proxy)  │◀────────────│                              │ │
│  └──────────┘             │  Auth gate → page router     │ │
│                           │  ├── Market Overview         │ │
│                           │  ├── Stock Analysis          │ │
│                           │  ├── Predictions             │ │
│                           │  ├── Setup / Refresh         │ │
│                           │  └── Admin Panel             │ │
│                           └──────────┬───────────────────┘ │
│                                      │                      │
│             ┌────────────────────────┼───────────────────┐  │
│             │                        │                   │  │
│      ┌──────▼──────┐    ┌────────────▼──────┐           │  │
│      │  src/        │    │  src/             │           │  │
│      │  predictor  │    │  scraper          │           │  │
│      │  .py        │    │  .py              │           │  │
│      └──────┬──────┘    └────────┬──────────┘           │  │
│             │                    │                       │  │
│      ┌──────▼────────────────────▼──────┐               │  │
│      │         src/database.py          │               │  │
│      │         SQLite: data/cse.db      │               │  │
│      └──────────────────────────────────┘               │  │
│                                                          │  │
│  ┌──────────────────────────────────────────────────┐   │  │
│  │  src/scheduler.py  (APScheduler, separate PID)   │   │  │
│  │  Mon–Fri 10:00 UTC → scraper + predictor         │   │  │
│  └──────────────────────────────────────────────────┘   │  │
│                                                          │  │
│  ┌────────────────────────────────┐                      │  │
│  │  models/  (one .pkl per stock) │                      │  │
│  └────────────────────────────────┘                      │  │
└─────────────────────────────────────────────────────────────┘
         │                                  ▲
         │  HTTPS                           │ JWT cookie
         ▼                                  │
┌─────────────────┐                ┌─────────────────────┐
│  cse.lk API     │                │  Google OAuth 2.0   │
│  /api/charts    │                │  accounts.google.com│
│  /allSecurity   │                └─────────────────────┘
│  Code           │
└─────────────────┘
```

---

## 3. Repository Layout

```
cse/
├── app.py                     # Streamlit entry point — auth gate + 5 pages
├── config.py                  # Secrets: CSE token, Google OAuth, admin emails
├── requirements.txt           # Python dependencies (version lower-bounds)
├── deploy.py                  # AWS EC2 lifecycle automation (boto3 + SCP)
│
├── src/
│   ├── __init__.py
│   ├── auth.py                # Google OAuth 2.0 flow + HMAC state verification
│   ├── database.py            # SQLite schema, migrations, all CRUD
│   ├── scraper.py             # CSE API client (session, auth, OHLCV fetch)
│   ├── predictor.py           # Feature engineering + Random Forest per symbol
│   └── scheduler.py          # APScheduler blocking job runner
│
├── deploy/
│   ├── server_setup.sh        # One-time EC2 bootstrap (apt, venv, nginx, systemd)
│   ├── cse-app.service        # systemd unit: Streamlit on 127.0.0.1:8501
│   ├── cse-scheduler.service  # systemd unit: daily scheduler process
│   └── cse-app-key.pem        # SSH private key (gitignored)
│
├── data/
│   └── cse.db                 # SQLite database (gitignored, lives on EBS)
│
└── models/
    └── <SYMBOL>.pkl           # Trained Pipeline per stock (gitignored, on EBS)
```

---

## 4. Data Flow

### 4.1 Bootstrap (first-time, manual)

```
operator
  └─▶ app.py Setup page  OR  py -c "bootstrap_all_securities(years=3)"
        └─▶ scraper.fetch_securities()
              GET https://www.cse.lk/api/allSecurityCode   → 319 symbols + names
              POST https://www.cse.lk/api/allSectors       → sector mapping
        └─▶ for each symbol:
              scraper.fetch_history(symbol, from, to)
                POST https://www.cse.lk/api/charts (form-encoded, JWT cookie)
                  body: symbol=X&fromDate=DD-MM-YYYY&toDate=DD-MM-YYYY&period=1
                  response: [{tradeDate(ms), open, high, low, close,
                              shareVolume, tradeVolume}, …]
              db.upsert_prices(records)   → daily_prices table
              sleep 1.2s   (rate limiting)
        └─▶ predictor.train_all()
              for each symbol:
                db.get_prices() → DataFrame
                _add_features() → MA5/10/20, RSI-14, MACD, Bollinger, vol ratio, DoW
                RandomForestClassifier(200 trees, max_depth=6).fit(80% split)
                joblib.dump → models/<SYMBOL>.pkl
        └─▶ predictor.predict_all()
              for each symbol: predict_next_day() → db.upsert_prediction()
```

### 4.2 Daily refresh (automated)

```
scheduler.py (systemd cse-scheduler.service, always running)
  └─▶ APScheduler CronTrigger  Mon–Fri 10:00 UTC (15:30 Sri Lanka time)
        └─▶ scraper.refresh_today()
              POST /api/tradeSummary or /api/tradeToTrade (today's date)
              db.upsert_prices(records)
        └─▶ predictor.predict_all()
              updates predictions table with tomorrow's forecasts
```

### 4.3 User login (Google OAuth 2.0 — authorization code flow)

```
Browser
  └─▶ GET / (Streamlit)
        auth.handle_oauth_callback() → no ?code= → shows login page
        user clicks "Sign in with Google"
          auth.google_login_url()
            state = HMAC_SHA256(CLIENT_SECRET, random_token)[:16]
            redirect to accounts.google.com/o/oauth2/v2/auth?code&state=...
  └─▶ Google consent screen
        user approves
        Google redirects to http://<EC2>/  with ?code=...&state=...
  └─▶ GET /?code=...&state=...  (Streamlit rerun)
        auth.handle_oauth_callback()
          _verify_state(state) → HMAC re-computation (no session needed)
          POST /oauth2/googleapis.com/token → access_token
          GET  /oauth2/v3/userinfo         → email, name, picture
          db.upsert_user() → role (admin / pending / approved)
          st.session_state["user"] = {...}
          st.session_state["role"] = role
          st.query_params.clear()
  └─▶ GET / (clean URL, authenticated session)
```

---

## 5. Authentication & Authorisation

### 5.1 Google OAuth CSRF protection

The standard approach of storing a nonce in `st.session_state` doesn't work in Streamlit because the session is destroyed when the browser navigates to Google and back. Instead, the state token is self-verifying:

```python
state = f"{random_token}.{HMAC_SHA256(CLIENT_SECRET, random_token)[:16]}"
```

On callback, the HMAC is re-computed and compared with `hmac.compare_digest` (constant-time). This requires no server-side storage.

### 5.2 Role model

| Role       | How assigned                                | Access                              |
|------------|---------------------------------------------|-------------------------------------|
| `admin`    | Email in `config.ADMIN_EMAILS` at login     | All pages + Admin Panel             |
| `approved` | Admin clicks Approve in Admin Panel         | Market Overview + Predictions       |
| `pending`  | Default for all new sign-ups                | "Awaiting approval" screen only     |
| `rejected` | Admin clicks Reject                         | Blocked                             |

The role is enforced in two places:
1. **Sidebar** — `page_options` list is filtered by role before rendering.
2. **Auth gate** — `role == "pending"` short-circuits to the pending screen before any page code runs.

---

## 6. Database

Single SQLite file at `data/cse.db`. All access goes through `src/database.py` — no raw SQL outside that module.

### Schema

```sql
-- Static reference data
CREATE TABLE securities (
    symbol      TEXT PRIMARY KEY,
    name        TEXT,
    sector      TEXT,
    updated_at  TEXT
);

-- OHLCV time series — core data store
CREATE TABLE daily_prices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,          -- YYYY-MM-DD
    open        REAL,                   -- nullable (CSE API returns null sometimes)
    high        REAL,
    low         REAL,
    close       REAL,
    volume      INTEGER,               -- shareVolume from CSE
    trades      INTEGER,               -- tradeVolume (number of trades)
    UNIQUE(symbol, date)
);
CREATE INDEX idx_prices_symbol_date ON daily_prices(symbol, date);

-- ML predictions (one row per symbol per future date)
CREATE TABLE predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    predicted_for   TEXT NOT NULL,     -- the date being predicted (tomorrow)
    direction       TEXT,              -- "UP" or "DOWN"
    confidence      REAL,              -- max(proba[0], proba[1])
    predicted_close REAL,              -- estimated price using avg daily move
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, predicted_for)
);

-- User registry (Google SSO)
CREATE TABLE users (
    email        TEXT PRIMARY KEY,
    name         TEXT,
    picture      TEXT,
    role         TEXT NOT NULL DEFAULT 'pending',
    created_at   TEXT DEFAULT (datetime('now')),
    approved_at  TEXT,
    approved_by  TEXT
);
```

### Upsert semantics

All writes use `INSERT … ON CONFLICT DO UPDATE` so re-running the scraper or bootstrapping again is always safe — no duplicate-key errors.

---

## 7. Scraper (`src/scraper.py`)

### Authentication strategy

The CSE API's `/api/charts` endpoint requires a JWT `accessToken` cookie. CSE uses an external identity provider (IdentityServer4 / Google social login), so standard credential-based ROPC login does not work reliably. The practical authentication path is:

1. Log into cse.lk in a browser.
2. Copy the `accessToken` cookie value (starts with `eyJ...`) from DevTools → Application → Cookies.
3. Paste into `config.py` as `CSE_TOKEN`. The token expires every ~3 hours.

The token is applied to a shared `requests.Session` as both a cookie and a `Bearer` authorization header. The Setup page in the dashboard allows pasting a fresh token without SSH access.

### API endpoints used

| Method | Path                          | Auth     | Purpose                        |
|--------|-------------------------------|----------|--------------------------------|
| GET    | `/api/allSecurityCode`        | No       | All 319 symbols + names        |
| POST   | `/api/allSectors`             | No       | Sector name → symbol mapping   |
| POST   | `/api/charts` (form-encoded)  | Yes      | OHLCV history for one symbol   |
| POST   | `/api/tradeSummary`           | No       | Daily summary (fallback used in `refresh_today`) |

### Rate limiting

A 1.2-second sleep between requests to avoid overwhelming the CSE API during bootstrap. Total bootstrap time for 3 years × 319 symbols ≈ 20–40 minutes.

---

## 8. ML Pipeline (`src/predictor.py`)

### Per-symbol model

One scikit-learn `Pipeline` per stock symbol, saved as `models/<SYMBOL>.pkl`. Training is triggered manually (Setup page) or can be automated.

### Feature engineering

All features are computed from the `close`, `volume`, and `date` columns of `daily_prices`:

| Feature          | Description                                          |
|------------------|------------------------------------------------------|
| `price_vs_ma5`   | `close / MA(5) - 1`                                  |
| `price_vs_ma10`  | `close / MA(10) - 1`                                 |
| `price_vs_ma20`  | `close / MA(20) - 1`                                 |
| `rsi`            | RSI-14 (Wilder's smoothing)                          |
| `macd_diff`      | MACD line minus signal line (EMA12 - EMA26, signal=EMA9) |
| `bb_position`    | Position within Bollinger Bands: `(close - mid) / (2σ)` |
| `vol_ratio`      | `volume / MA(volume, 10)`                            |
| `dow`            | Day-of-week (0=Mon … 4=Fri)                          |

**Target:** `1` if `close[t+1] > close[t]`, else `0`.

### Model

```
Pipeline([
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
])
```

- Minimum 60 trading days required to train.
- 80/20 chronological train/test split (no shuffle — time series integrity).
- Predicted close = `last_close × (1 ± avg_daily_abs_return)`.

---

## 9. Scheduler (`src/scheduler.py`)

APScheduler `BlockingScheduler` running in a dedicated OS process managed by `cse-scheduler.service`. It has a single job:

```
CronTrigger(day_of_week="mon-fri", hour=10, minute=0)  # 10:00 UTC = 15:30 SL
```

Job steps:
1. `scraper.refresh_today()` — fetch today's trade summary, upsert to DB.
2. `predictor.predict_all()` — generate predictions for tomorrow, upsert to DB.

The process is set to `Restart=always` with a 30-second backoff in systemd, so a crash (e.g. network timeout) self-heals.

---

## 10. Infrastructure

### EC2 instance

| Property         | Value                                         |
|------------------|-----------------------------------------------|
| Type             | t2.micro (1 vCPU, 1 GiB RAM)                 |
| AMI              | Ubuntu 22.04 LTS (Canonical, x86_64)          |
| Region           | ap-southeast-1 (Singapore)                    |
| Storage          | EBS root volume (db + models persist here)    |
| Ports open       | 22 (SSH), 80 (HTTP)                           |
| Public hostname  | `ec2-122-248-216-100.ap-southeast-1.compute.amazonaws.com` |

### Process layout

```
systemd
  ├── nginx.service           — reverse proxy, port 80 → 127.0.0.1:8501
  ├── cse-app.service         — streamlit run app.py --server.port=8501
  └── cse-scheduler.service   — python -m src.scheduler
```

Nginx handles WebSocket upgrade headers required by Streamlit's internal protocol (`Upgrade: websocket`, `Connection: upgrade`) and sets `proxy_read_timeout 86400` to keep long-lived connections alive.

### Deployment automation (`deploy.py`)

| Command              | Effect                                                          |
|----------------------|-----------------------------------------------------------------|
| `py deploy.py setup` | Provision EC2 (key pair, SG, AMI lookup, launch, SSH wait), run `server_setup.sh`, deploy code, start services |
| `py deploy.py deploy`| rsync (falls back to SCP on Windows) + chown + systemctl restart |
| `py deploy.py token` | `sed` the `CSE_TOKEN` line in `config.py` on server + restart  |
| `py deploy.py status`| `systemctl status cse-app cse-scheduler`                       |
| `py deploy.py ssh`   | Print / open SSH session                                        |

Files excluded from deployment: `data/`, `models/`, `__pycache__`, `.git`, `.github`, `*.pyc`, `notebooks/`, `tests/`, `deploy/`, `.gitignore`.

---

## 11. Configuration & Secrets

All secrets live in `config.py` (gitignored). Never committed.

| Variable              | Purpose                                          |
|-----------------------|--------------------------------------------------|
| `CSE_EMAIL`           | CSE account (kept for reference)                 |
| `CSE_PASSWORD`        | CSE account (ROPC fallback, rarely succeeds)     |
| `CSE_TOKEN`           | JWT accessToken from browser cookie; expires ~3h |
| `GOOGLE_CLIENT_ID`    | Google Cloud OAuth 2.0 client ID                 |
| `GOOGLE_CLIENT_SECRET`| Google Cloud OAuth 2.0 client secret (also used as HMAC key for state signing) |
| `REDIRECT_URI`        | Registered Google redirect URI (must include trailing `/`) |
| `ADMIN_EMAILS`        | List of emails that always receive `admin` role  |

---

## 12. Key Design Decisions & Trade-offs

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| SQLite over PostgreSQL/RDS | Zero ops cost, no network hop, survives EC2 restarts on EBS | Single writer, no horizontal scale |
| One .pkl model per symbol | Independent training schedules, no symbol interferes with another | ~319 files on disk, ~50–200 KB each |
| Session-free HMAC state for OAuth | `st.session_state` is lost across browser redirects; HMAC avoids server storage | `GOOGLE_CLIENT_SECRET` must be stable; rotating it invalidates any in-flight login |
| Streamlit over Flask/FastAPI | No-code UI iteration, built-in widgets for charts and tables | Limited multi-page routing, no fine-grained HTTP control |
| t2.micro free tier | Zero cost for 12 months | 1 GiB RAM — bootstrap may be slow; ML training for 319 symbols fits but is tight |
| Manual token refresh | CSE uses external IdP (IdentityServer4); ROPC grant unreliable | Operator must paste a fresh token every ~3 hours during bootstrap |
| SCP fallback on Windows | `rsync` is not shipped on Windows | Slower than rsync; copies all non-excluded files individually each deploy |

---

## 13. Known Limitations

- **Token expiry during bootstrap** — the CSE accessToken expires every ~3 hours. Bootstrap for 3 years of data takes 20–40 minutes, so at least one manual token refresh is typically needed. The Setup page and `py deploy.py token` command exist to make this as easy as possible.
- **Sector mapping incomplete** — the `/api/allSectors` endpoint returns sector data but the symbol-to-sector mapping format has not been fully confirmed; most stocks show `sector: Unknown`.
- **No HTTPS** — the EC2 instance serves HTTP only. Adding HTTPS requires either a custom domain with a TLS certificate (Let's Encrypt + nginx config) or an AWS ACM certificate with a Load Balancer.
- **Single-instance SQLite** — concurrent writes from the Streamlit app and the scheduler are serialized by SQLite's WAL mode implicitly but are not explicitly coordinated. Under normal operation (scheduler writes once a day, Streamlit writes only during manual refreshes) this is not an issue.
- **Prediction accuracy computed on-the-fly** — historical accuracy is derived by joining `predictions` with `daily_prices` at query time rather than being stored. The correlated subquery for "previous trading day" may be slow on very large datasets (>100 k rows); add a materialised view or a `prev_close` column if performance degrades.

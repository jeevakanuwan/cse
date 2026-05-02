# CSE Stock Prediction — Project Reference

## Purpose
Accumulate historical trading data from the Colombo Stock Exchange (CSE) and predict next-day price direction (UP/DOWN) for all ~319 listed securities.

## Stack
- **Streamlit** — dashboard UI (`streamlit run app.py`)
- **SQLite** — local database at `data/cse.db`
- **scikit-learn** — Random Forest classifier for next-day prediction
- **Python 3.14** — runtime (`py` command on Windows)

## Project Structure
```
cse/
├── app.py              # Streamlit dashboard — 4 pages
├── config.py           # CSE credentials + access token (DO NOT COMMIT)
├── deploy.py           # AWS deployment automation script
├── requirements.txt
├── deploy/
│   ├── server_setup.sh       # One-time EC2 server setup
│   ├── cse-app.service       # systemd: Streamlit dashboard
│   └── cse-scheduler.service # systemd: daily data fetch
├── src/
│   ├── database.py     # SQLite schema + CRUD
│   ├── scraper.py      # CSE API scraper
│   ├── predictor.py    # Feature engineering + Random Forest
│   └── scheduler.py    # Daily auto-fetch (runs after market close)
├── data/
│   └── cse.db          # SQLite database (gitignored)
└── models/             # Saved .pkl model files per symbol
```

## How to Run (Local)

### Install dependencies (once)
```cmd
py -m pip install -r requirements.txt
```

### First-time data bootstrap (~20-40 min)
```cmd
py -c "from src.scraper import bootstrap_all_securities; bootstrap_all_securities(years=3)"
```

### Launch dashboard
```cmd
streamlit run app.py
```

### Daily auto-fetch (run in background, keep open)
```cmd
py -m src.scheduler
```

---

## AWS Deployment

### Architecture
- **EC2 t2.micro** — free tier eligible (12 months), Ubuntu 22.04
- **Region**: ap-southeast-1 (Singapore — lowest latency to Sri Lanka)
- **Ports**: 22 (SSH), 80 (web via nginx → Streamlit)
- **Storage**: EBS root volume (SQLite db lives here, persists across restarts)
- Two systemd services: one for the Streamlit app, one for the daily scheduler

### One-time AWS account setup (do this once)

**Step 1 — Create an AWS account**
Go to https://aws.amazon.com → Create account. You need a credit card but t2.micro is free for 12 months.

**Step 2 — Create an IAM user (safer than using root)**
1. AWS Console → IAM → Users → Create user
2. Username: `cse-deploy`
3. Attach policy: **AmazonEC2FullAccess**
4. Create user → Security credentials tab → Create access key → CLI use case
5. Download the CSV — save the Access Key ID and Secret Access Key

**Step 3 — Install AWS CLI**
Download from: https://aws.amazon.com/cli/
After install, run:
```cmd
aws configure
```
Enter your Access Key ID, Secret Access Key, region `ap-southeast-1`, output format `json`.

**Step 4 — Install deploy dependencies**
```cmd
py -m pip install boto3 paramiko
```

### Deploy commands

```cmd
# First-time: create EC2 instance and set everything up (~5 min)
py deploy.py setup

# Push code updates to server
py deploy.py deploy

# Update CSE token on server (do this when token expires ~every 3h during bootstrap)
py deploy.py token

# Check if app is running on server
py deploy.py status

# Open SSH session directly to the server
py deploy.py ssh
```

### After deployment
- Dashboard available at: `http://<your-ec2-ip>` (port 80)
- SSH key saved at: `deploy/cse-app-key.pem`
- Server app directory: `/opt/cse-app/`
- To view logs on server: `journalctl -u cse-app -f`

### Token refresh on server
The CSE accessToken expires every ~3 hours. When it does:
1. Get a fresh token from your browser (F12 → Application → Cookies → cse.lk → accessToken)
2. Paste into `config.py` as `CSE_TOKEN`
3. Run `py deploy.py token` — pushes it to the server automatically

---

## Authentication & User Roles

### Roles
| Role | Access |
|------|--------|
| `admin` | All pages + Admin Panel (approve/reject users) |
| `approved` | Market Overview + Predictions only |
| `pending` | "Awaiting approval" screen only |
| `rejected` | Blocked |

Admin emails are set in `config.py → ADMIN_EMAILS`. These always get admin access on first login.

### Google OAuth Setup (one-time)

**Step 1 — Create a Google Cloud project**
1. Go to https://console.cloud.google.com
2. New project → name it `CSE Dashboard`

**Step 2 — Create OAuth credentials**
1. APIs & Services → Credentials → Create Credentials → OAuth client ID
2. Application type: **Web application**
3. Name: `CSE Dashboard`
4. Authorized redirect URIs — add both:
   - `http://localhost:8501` (local dev)
   - `http://your-ec2-ip` (production, add after deploying)
5. Click Create → copy **Client ID** and **Client Secret**

**Step 4 — Configure OAuth consent screen**
1. APIs & Services → OAuth consent screen
2. User type: **External**
3. App name: `CSE Dashboard`, support email: your email
4. Scopes: add `email`, `profile`, `openid`
5. Save

**Step 5 — Paste into config.py**
```python
GOOGLE_CLIENT_ID     = "xxxx.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "GOCSPX-xxxx"
REDIRECT_URI         = "http://localhost:8501"
ADMIN_EMAILS         = ["jeevakanuwan@gmail.com"]
```

### How approval works
1. User visits app → clicks **Sign in with Google**
2. Google redirects back → user is created in DB with role `pending`
3. Admin sees them in **Admin Panel** → clicks **Approve**
4. User now has access to Market Overview + Predictions
5. Admin can promote any user to admin or revoke access at any time

---

## CSE API — Confirmed Endpoints

All base URL: `https://www.cse.lk/api`

| Method | Path | Content-Type | Body | Auth | Purpose |
|--------|------|-------------|------|------|---------|
| GET | `/allSecurityCode` | — | — | No | All 319 securities (symbol + name) |
| POST | `/allSectors` | application/json | `{"period":"4"}` | No | Sector list |
| POST | `/charts` | application/x-www-form-urlencoded | `symbol=X&fromDate=DD-MM-YYYY&toDate=DD-MM-YYYY&period=1` | **Yes** | OHLCV history |

### /api/charts Response Format
```json
{
  "open": 20.3,
  "high": 20.5,
  "low": 20.1,
  "close": 20.2,
  "turnover": 191992234.3,
  "shareVolume": 9494513,
  "tradeVolume": 427,
  "tradeDate": 1743100200000
}
```
- `tradeDate` is **milliseconds** since epoch (divide by 1000 for Unix timestamp)
- `open` / `low` can be `null` on some days
- `shareVolume` = share count traded
- `tradeVolume` = number of trades

## Authentication

The `/api/charts` endpoint requires a JWT `accessToken` cookie.
CSE uses an external identity provider (Google/social) — simple email+password POST does not work.

### How to get the token
1. Log in to https://www.cse.lk in browser
2. F12 → Application → Cookies → `https://www.cse.lk`
3. Copy the `accessToken` cookie value (starts with `eyJ...`)
4. Paste into `config.py` as `CSE_TOKEN`

**Token expires every ~3 hours.** Refresh from browser when it expires.

### config.py format
```python
CSE_EMAIL    = "jeevakanuwan@gmail.com"
CSE_PASSWORD = "..."          # kept for reference, not used for auth
CSE_TOKEN    = "eyJ..."       # paste fresh token here when expired
```

## ML Model

- **Algorithm**: Random Forest classifier (200 trees, max_depth=6)
- **Features**: MA5/10/20 ratios, RSI-14, MACD diff, Bollinger Band position, volume ratio, day-of-week
- **Target**: Next-day direction (1 = UP, 0 = DOWN/flat)
- **Min data**: 60 trading days required to train
- **Output**: direction (UP/DOWN) + confidence % + predicted close price
- **Models saved**: `models/<symbol>.pkl` — one file per stock

## Database Schema

```sql
securities   (symbol PK, name, sector, updated_at)
daily_prices (symbol, date, open, high, low, close, volume, trades) UNIQUE(symbol,date)
predictions  (symbol, predicted_for, direction, confidence, predicted_close) UNIQUE(symbol,predicted_for)
```

## Dashboard Pages
1. **Market Overview** — gainers, losers, full table with search
2. **Stock Analysis** — candlestick + MA chart + next-day prediction for any stock (search by symbol or company name)
3. **Predictions** — all 319 stocks predicted UP/DOWN with confidence, searchable by name
4. **Setup / Refresh** — bootstrap, manual refresh, retrain buttons

## CSE Market Hours
- Trading: 09:30 – 14:30 Sri Lanka Time (UTC+5:30)
- Scheduler fetches at 15:30 SL = 10:00 UTC (Mon–Fri)

## Known Issues / Notes
- `allSectors` endpoint returns sector data but sector-to-symbol mapping not yet confirmed — most stocks show `sector: Unknown`
- Token refresh must be done manually every ~3 hours during bootstrap
- `open` and `low` fields are occasionally `null` in CSE API responses — stored as NULL in DB, handled gracefully by predictor

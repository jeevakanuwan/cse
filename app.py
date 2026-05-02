"""
Colombo Stock Exchange — Historical Trends & Prediction Dashboard
Run with:  streamlit run app.py
"""

import logging
import re
import sys
import base64
import json as _json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src import database as db
from src import scraper, predictor
from src import auth

logging.basicConfig(level=logging.INFO)

db.init_db()

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CSE Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Login / pending pages (defined before auth gate so they're available)
# ---------------------------------------------------------------------------

def _show_login():
    st.markdown("""
    <style>
    .login-box {
        max-width: 420px; margin: 80px auto; padding: 48px 40px;
        background: #1e1e2e; border-radius: 16px;
        box-shadow: 0 4px 32px rgba(0,0,0,0.4); text-align: center;
    }
    .login-title { font-size: 2rem; font-weight: 700; margin-bottom: 4px; }
    .login-sub   { color: #888; margin-bottom: 32px; font-size: 0.95rem; }
    </style>
    <div class="login-box">
        <div class="login-title">📈 CSE Dashboard</div>
        <div class="login-sub">Colombo Stock Exchange · Predictions & Trends</div>
    </div>
    """, unsafe_allow_html=True)

    if not auth.is_configured():
        st.warning(
            "Google OAuth is not configured yet. "
            "Add `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and `REDIRECT_URI` to `config.py`."
        )
        return

    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.link_button(
            "🔑  Sign in with Google",
            url=auth.google_login_url(),
            use_container_width=True,
            type="primary",
        )
        st.caption("Access is by approval only. Sign in to request access.")


def _show_pending(user: dict):
    st.markdown("---")
    st.markdown(f"## Welcome, {user['name']} 👋")
    st.info(
        "Your account is **pending approval**.  \n"
        "An admin will review your request shortly. "
        "You'll have full access once approved."
    )
    st.markdown(f"Signed in as: `{user['email']}`")
    if st.button("Sign out"):
        auth.logout()
        st.rerun()


# ---------------------------------------------------------------------------
# Auth gate — runs before anything else
# ---------------------------------------------------------------------------

auth.handle_oauth_callback()   # detect ?code=... from Google redirect
user = auth.get_user()
role = auth.get_role()

if user is None:
    _show_login()
    st.stop()

if role == "pending":
    _show_pending(user)
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar (only shown to authenticated users)
# ---------------------------------------------------------------------------

with st.sidebar:
    # User avatar + name
    if user.get("picture"):
        st.image(user["picture"], width=48)
    st.markdown(f"**{user['name']}**  \n{user['email']}")
    if role == "admin":
        st.caption("🛡️ Admin")
    st.markdown("---")

    # Role-based page list
    if role == "admin":
        page_options = [
            "Market Overview",
            "Stock Analysis",
            "Predictions",
            "Setup / Refresh",
            "Admin Panel",
        ]
    else:
        page_options = ["Market Overview", "Predictions"]

    page = st.radio("Navigate", page_options)
    st.markdown("---")

    if st.button("Sign out"):
        auth.logout()
        st.rerun()

    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_market_summary():
    return db.get_market_summary()


@st.cache_data(ttl=300)
def load_securities():
    return db.get_all_securities()


@st.cache_data(ttl=300)
def load_predictions():
    return db.get_latest_predictions()


@st.cache_data(ttl=60)
def load_prices(symbol: str, days: int):
    return db.get_prices(symbol, limit=days)


def direction_badge(direction: str) -> str:
    return "🟢 UP" if direction == "UP" else "🔴 DOWN"


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    df = df.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name=symbol,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"], name="Volume",
        marker_color="rgba(100,100,200,0.35)", yaxis="y2",
    ))
    fig.update_layout(
        title=f"{symbol} — Price & Volume", xaxis_title="Date",
        yaxis_title="Price (LKR)",
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        xaxis_rangeslider_visible=False, template="plotly_dark",
        height=480, margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def moving_avg_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    df = df.sort_values("date").copy()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close",  line=dict(color="#90caf9")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA20"],  name="MA 20",  line=dict(color="#ffb74d", dash="dash")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA50"],  name="MA 50",  line=dict(color="#ef9a9a", dash="dot")))
    fig.update_layout(
        title=f"{symbol} — Moving Averages", xaxis_title="Date",
        yaxis_title="Price (LKR)", template="plotly_dark",
        height=380, margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_market_overview():
    st.title("Market Overview")
    summary = load_market_summary()

    if summary.empty:
        st.warning("No market data found. Go to **Setup / Refresh** to load data for the first time.")
        return

    latest_date = summary["date"].max() if "date" in summary.columns else "—"
    st.caption(f"Latest data: {latest_date}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Listed Securities", len(summary))
    c2.metric("Gainers", int((summary["change_pct"] > 0).sum()))
    c3.metric("Losers",  int((summary["change_pct"] < 0).sum()))
    c4.metric("Avg Volume", f"{summary['volume'].mean():,.0f}")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🟢 Top Gainers", "🔴 Top Losers", "All Securities"])

    with tab1:
        _show_table(summary[summary["change_pct"] > 0].nlargest(20, "change_pct"))
    with tab2:
        _show_table(summary[summary["change_pct"] < 0].nsmallest(20, "change_pct"))
    with tab3:
        search = st.text_input("Search symbol or name", "")
        filtered = summary
        if search:
            mask = (
                summary["symbol"].str.contains(search, case=False, na=False) |
                summary["name"].str.contains(search, case=False, na=False)
            )
            filtered = summary[mask]
        _show_table(filtered)


def _show_table(df: pd.DataFrame):
    if df.empty:
        st.info("No data.")
        return
    display = df[["symbol", "name", "sector", "close", "open", "high", "low", "volume", "change_pct"]].copy()
    display.columns = ["Symbol", "Name", "Sector", "Close", "Open", "High", "Low", "Volume", "Change %"]
    display["Change %"] = display["Change %"].apply(
        lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%" if pd.notna(x) else "—"
    )
    st.dataframe(display.reset_index(drop=True), use_container_width=True)


def page_stock_analysis():
    st.title("Stock Analysis")
    securities = load_securities()
    if securities.empty:
        st.warning("No securities loaded yet. Go to **Setup / Refresh** first.")
        return

    securities["label"] = securities["symbol"] + " — " + securities["name"]
    label_to_symbol = dict(zip(securities["label"], securities["symbol"]))
    selected_label = st.selectbox("Search by symbol or company name", securities["label"].tolist())
    symbol = label_to_symbol[selected_label]

    days_map = {"3 months": 90, "6 months": 180, "1 year": 365, "2 years": 730}
    days = days_map[st.select_slider("Period", options=list(days_map.keys()), value="1 year")]

    df = load_prices(symbol, days)
    if df.empty:
        st.info(f"No price data for {symbol}.")
        return

    df = df.sort_values("date")
    last = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Close",  f"LKR {last['close']:.2f}")
    c2.metric("High",   f"LKR {last['high']:.2f}")
    c3.metric("Low",    f"LKR {last['low']:.2f}")
    c4.metric("Volume", f"{int(last['volume']):,}")

    st.plotly_chart(candlestick_chart(df, symbol), use_container_width=True)
    st.plotly_chart(moving_avg_chart(df, symbol),  use_container_width=True)

    st.markdown("### Next-Day Prediction")
    with st.spinner("Running prediction …"):
        result = predictor.predict_next_day(symbol)

    if result:
        p1, p2, p3 = st.columns(3)
        p1.metric("Direction",       direction_badge(result["direction"]))
        p2.metric("Confidence",      f"{result['confidence']*100:.1f}%")
        p3.metric("Predicted Close", f"LKR {result['predicted_close']:.2f}")
    else:
        st.info("Not enough historical data to generate a prediction (need ≥ 60 trading days).")

    with st.expander("Raw price data"):
        st.dataframe(df.sort_values("date", ascending=False).reset_index(drop=True), use_container_width=True)


def page_predictions():
    st.title("Predictions — All Securities")
    preds = load_predictions()

    if preds.empty:
        st.warning("No predictions yet.")
        if st.button("Run predictions for all securities now"):
            with st.spinner("Running predictions …"):
                predictor.predict_all()
            st.success("Done! Refresh the page.")
        return

    predicted_for = preds["predicted_for"].iloc[0]
    st.caption(f"Predictions for: **{predicted_for}**")

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Predicted UP",   int((preds["direction"] == "UP").sum()))
    c2.metric("Predicted DOWN", int((preds["direction"] == "DOWN").sum()))
    search = c3.text_input("Search by symbol or company name", "")

    if search:
        mask = (
            preds["symbol"].str.contains(search, case=False, na=False) |
            preds["name"].str.contains(search, case=False, na=False)
        )
        preds = preds[mask]

    cols = ["symbol", "name", "sector", "direction", "confidence", "predicted_close"]
    tab1, tab2 = st.tabs(["🟢 Predicted UP", "🔴 Predicted DOWN"])
    with tab1:
        _pred_table(preds[preds["direction"] == "UP"].sort_values("confidence", ascending=False)[cols])
    with tab2:
        _pred_table(preds[preds["direction"] == "DOWN"].sort_values("confidence", ascending=False)[cols])


def _pred_table(df: pd.DataFrame):
    if df.empty:
        st.info("No data.")
        return
    display = df.copy()
    display["confidence"] = display["confidence"].apply(lambda x: f"{x*100:.1f}%")
    display.columns = ["Symbol", "Name", "Sector", "Direction", "Confidence", "Predicted Close"]
    st.dataframe(display.reset_index(drop=True), use_container_width=True)


def page_admin():
    st.title("Admin Panel — User Management")

    users_df = db.get_all_users()
    if users_df.empty:
        st.info("No users yet.")
        return

    pending  = users_df[users_df["role"] == "pending"]
    approved = users_df[users_df["role"] == "approved"]
    admins   = users_df[users_df["role"] == "admin"]

    # ── Pending approvals ────────────────────────────────────────────────────
    st.markdown(f"### ⏳ Pending Approvals ({len(pending)})")
    if pending.empty:
        st.success("No pending users.")
    else:
        for _, row in pending.iterrows():
            c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
            c1.write(f"**{row['name']}**  \n{row['email']}")
            c2.caption(f"Signed up: {row['created_at'][:10]}")
            if c3.button("✅ Approve", key=f"approve_{row['email']}"):
                db.set_user_role(row["email"], "approved", user["email"])
                st.rerun()
            if c4.button("❌ Reject", key=f"reject_{row['email']}"):
                db.set_user_role(row["email"], "rejected", user["email"])
                st.rerun()

    st.markdown("---")

    # ── Approved subscribers ─────────────────────────────────────────────────
    st.markdown(f"### ✅ Approved Subscribers ({len(approved)})")
    if approved.empty:
        st.info("No approved subscribers yet.")
    else:
        for _, row in approved.iterrows():
            c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
            c1.write(f"**{row['name']}**  \n{row['email']}")
            c2.caption(f"Approved: {(row['approved_at'] or '')[:10]}")
            if c3.button("🛡️ Make Admin", key=f"mkadmin_{row['email']}"):
                db.set_user_role(row["email"], "admin", user["email"])
                st.rerun()
            if c4.button("🚫 Revoke", key=f"revoke_{row['email']}"):
                db.set_user_role(row["email"], "pending", user["email"])
                st.rerun()

    st.markdown("---")

    # ── Admins ───────────────────────────────────────────────────────────────
    st.markdown(f"### 🛡️ Admins ({len(admins)})")
    for _, row in admins.iterrows():
        c1, c2, c3 = st.columns([3, 3, 1])
        c1.write(f"**{row['name']}**  \n{row['email']}")
        c2.caption("Full access")
        # prevent self-demotion
        if row["email"] != user["email"]:
            if c3.button("↓ Demote", key=f"demote_{row['email']}"):
                db.set_user_role(row["email"], "approved", user["email"])
                st.rerun()


def page_setup():
    st.title("Setup & Data Refresh")

    # ── Token management ─────────────────────────────────────────────────────
    st.markdown("### 🔑 CSE Login Token")
    st.markdown(
        "The CSE token expires every **~3 hours**.\n\n"
        "**How to get it:** Log in at [cse.lk](https://www.cse.lk) → "
        "`F12` → **Application** → **Cookies** → copy **`accessToken`**"
    )
    token_input = st.text_area("Paste accessToken here", height=80, placeholder="eyJ...")
    if st.button("💾 Save Token", type="primary"):
        token = token_input.split(";")[0].strip()
        if not token or not token.startswith("eyJ"):
            st.error("Invalid token. Copy the full value starting with eyJ.")
        else:
            config_path = Path(__file__).parent / "config.py"
            try:
                original = config_path.read_text()
                updated = re.sub(r'CSE_TOKEN\s*=\s*"[^"]*"', f'CSE_TOKEN = "{token}"', original)
                if "CSE_TOKEN" not in original:
                    updated = original.rstrip() + f'\nCSE_TOKEN = "{token}"\n'
                config_path.write_text(updated)
                from src.scraper import SESSION
                SESSION.cookies.set("accessToken", token, domain="www.cse.lk")
                SESSION.headers.update({"authorization": f"Bearer {token}"})
                import src.scraper as _sc
                _sc._logged_in = True
                st.success("Token saved and applied.")
            except Exception as e:
                st.error(f"Could not save token: {e}")

    # Token expiry status
    try:
        import config as _cfg
        tok = getattr(_cfg, "CSE_TOKEN", "").split(";")[0].strip()
        if tok and tok.startswith("eyJ"):
            payload = tok.split(".")[1] + "=="
            decoded = _json.loads(base64.urlsafe_b64decode(payload))
            exp = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)
            remaining = exp - datetime.now(tz=timezone.utc)
            if remaining.total_seconds() > 0:
                st.success(f"Token valid — expires in **{int(remaining.total_seconds()//60)} min**")
            else:
                st.warning("Token **expired** — paste a fresh one above.")
        else:
            st.info("No token saved yet.")
    except Exception:
        pass

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        years = st.slider("Years of history to download", 1, 5, 3)
        if st.button("🚀 Bootstrap (first-time setup)", type="primary"):
            st.warning("This will take 20–40 minutes.")
            progress = st.progress(0, text="Starting …")
            securities = scraper.fetch_securities()
            total = len(securities)
            if not securities:
                st.error("Could not fetch securities from CSE.")
            else:
                for i, sec in enumerate(securities):
                    db.upsert_security(sec["symbol"], sec["name"], sec["sector"])
                    from datetime import timedelta
                    to_date   = datetime.today().strftime("%Y-%m-%d")
                    from_date = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
                    records = scraper.fetch_history(sec["symbol"], from_date, to_date)
                    if records:
                        db.upsert_prices(records)
                    progress.progress((i + 1) / total, text=f"{i+1}/{total}: {sec['symbol']}")
                st.success(f"Bootstrap complete — {total} securities loaded.")

    with col2:
        st.markdown("#### Refresh today's data")
        if st.button("🔄 Fetch today's prices"):
            with st.spinner("Fetching …"):
                scraper.refresh_today()
            st.success("Today's prices updated.")
            st.cache_data.clear()

    with col3:
        st.markdown("#### Retrain & predict")
        if st.button("🤖 Retrain all models + predict"):
            with st.spinner("Training …"):
                predictor.train_all()
                predictor.predict_all()
            st.success("All models retrained and predictions saved.")
            st.cache_data.clear()

    st.markdown("---")
    st.markdown("#### Database stats")
    sec_df = db.get_all_securities()
    st.write(f"**Securities in DB:** {len(sec_df)}")
    try:
        with db.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
            last  = conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()[0]
        st.write(f"**Price records:** {count:,}")
        st.write(f"**Latest date:** {last or '—'}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "Market Overview":
    page_market_overview()
elif page == "Stock Analysis":
    page_stock_analysis()
elif page == "Predictions":
    page_predictions()
elif page == "Setup / Refresh":
    page_setup()
elif page == "Admin Panel":
    page_admin()

"""
Colombo Stock Exchange — Historical Trends & Prediction Dashboard
Run with:  streamlit run app.py
"""

import logging
import re
import sys
import base64
import json as _json
from datetime import datetime, timedelta, timezone
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
            "Predicted vs Actual",
            "Setup / Refresh",
            "Admin Panel",
        ]
    else:
        page_options = ["Market Overview", "Predictions", "Predicted vs Actual"]

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
def load_predictions(horizon: int = 1):
    return db.get_latest_predictions(horizon)


@st.cache_data(ttl=60)
def load_prices(symbol: str, days: int):
    return db.get_prices(symbol, limit=days)


@st.cache_data(ttl=300)
def load_prediction_history(horizon: int = 1):
    return db.get_prediction_history(horizon=horizon)


@st.cache_data(ttl=300)
def load_symbol_prediction_history(symbol: str, horizon: int = 1):
    return db.get_prediction_history(symbol=symbol, horizon=horizon)


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

    st.markdown("### Price Direction Forecast")
    horizon = st.select_slider(
        "Forecast horizon",
        options=list(range(1, 8)),
        value=1,
        format_func=lambda n: f"{n} day ahead" if n == 1 else f"{n} days ahead",
        key="analysis_horizon",
    )
    with st.spinner(f"Running {horizon}-day prediction …"):
        result = predictor.predict_next_day(symbol, horizon)

    if result:
        target_date = (datetime.today() + timedelta(days=horizon)).strftime("%Y-%m-%d")
        st.caption(f"Forecast for **{target_date}** ({horizon}-day horizon)")
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

    # Horizon selector — shared across both tabs
    horizon = st.select_slider(
        "Forecast horizon",
        options=list(range(1, 8)),
        value=1,
        format_func=lambda n: f"{n} day" if n == 1 else f"{n} days",
    )

    tab_today, tab_history = st.tabs(["🔮 Predictions", "📊 Accuracy History"])

    with tab_today:
        preds = load_predictions(horizon)
        if preds.empty:
            st.warning(f"No predictions for the {horizon}-day horizon yet.")
            if st.button("Generate predictions now", key="gen_preds"):
                with st.spinner(f"Running {horizon}-day predictions for all securities …"):
                    predictor.predict_all(horizon)
                st.success("Done! Refresh the page.")
                st.cache_data.clear()
        else:
            predicted_for = preds["predicted_for"].iloc[0]
            st.caption(f"Predicting price direction for: **{predicted_for}** ({horizon}-day horizon)")

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
            sub1, sub2 = st.tabs(["🟢 Predicted UP", "🔴 Predicted DOWN"])
            with sub1:
                _pred_table(preds[preds["direction"] == "UP"].sort_values("confidence", ascending=False)[cols])
            with sub2:
                _pred_table(preds[preds["direction"] == "DOWN"].sort_values("confidence", ascending=False)[cols])

    with tab_history:
        _show_prediction_history(horizon)


def _pred_table(df: pd.DataFrame):
    if df.empty:
        st.info("No data.")
        return
    display = df.copy()
    display["confidence"] = display["confidence"].apply(lambda x: f"{x*100:.1f}%")
    display.columns = ["Symbol", "Name", "Sector", "Direction", "Confidence", "Predicted Close"]
    st.dataframe(display.reset_index(drop=True), use_container_width=True)


def _show_prediction_history(horizon: int = 1):
    hist = load_prediction_history(horizon)

    if hist.empty:
        st.info(
            "No past predictions to evaluate yet. "
            "History appears once the scheduler has run at least one full day cycle."
        )
        return

    total      = len(hist)
    accuracy   = hist["correct"].mean() * 100
    up_hist    = hist[hist["predicted_direction"] == "UP"]
    down_hist  = hist[hist["predicted_direction"] == "DOWN"]
    up_acc     = up_hist["correct"].mean() * 100   if not up_hist.empty   else 0.0
    down_acc   = down_hist["correct"].mean() * 100 if not down_hist.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predictions Evaluated", f"{total:,}")
    c2.metric("Overall Accuracy",      f"{accuracy:.1f}%")
    c3.metric("UP Accuracy",           f"{up_acc:.1f}%")
    c4.metric("DOWN Accuracy",         f"{down_acc:.1f}%")

    st.markdown("---")

    # Monthly accuracy trend
    monthly = (
        hist.assign(month=hist["predicted_for"].str[:7])
        .groupby("month")["correct"]
        .agg(accuracy=lambda x: round(x.mean() * 100, 1), count="count")
        .reset_index()
    )
    monthly = monthly[monthly["count"] >= 5]

    if not monthly.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["month"],
            y=monthly["accuracy"],
            mode="lines+markers",
            name="Accuracy %",
            line=dict(color="#26a69a", width=2),
            marker=dict(size=7),
            hovertemplate="%{x}: %{y:.1f}%<br>(%{customdata} predictions)<extra></extra>",
            customdata=monthly["count"],
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="#888",
                      annotation_text="50% baseline", annotation_position="bottom right")
        fig.update_layout(
            title="Monthly Prediction Accuracy",
            xaxis_title="Month",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 100]),
            template="plotly_dark",
            height=340,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Per-Symbol Accuracy")

    sym_search = st.text_input("Search symbol or name", "", key="hist_sym_search")
    by_sym = (
        hist.groupby(["symbol", "name"])
        .agg(
            evaluated=("correct", "count"),
            accuracy=("correct", lambda x: round(x.mean() * 100, 1)),
            up_accuracy=("correct", lambda x: round(
                x[hist.loc[x.index, "predicted_direction"] == "UP"].mean() * 100, 1
            ) if (hist.loc[x.index, "predicted_direction"] == "UP").any() else float("nan")),
            down_accuracy=("correct", lambda x: round(
                x[hist.loc[x.index, "predicted_direction"] == "DOWN"].mean() * 100, 1
            ) if (hist.loc[x.index, "predicted_direction"] == "DOWN").any() else float("nan")),
            avg_confidence=("confidence", lambda x: round(x.mean() * 100, 1)),
        )
        .reset_index()
        .sort_values("evaluated", ascending=False)
    )

    if sym_search:
        mask = (
            by_sym["symbol"].str.contains(sym_search, case=False, na=False) |
            by_sym["name"].str.contains(sym_search, case=False, na=False)
        )
        by_sym = by_sym[mask]

    by_sym_display = by_sym.copy()
    by_sym_display.columns = [
        "Symbol", "Name", "Evaluated", "Accuracy %",
        "UP Accuracy %", "DOWN Accuracy %", "Avg Confidence %",
    ]
    st.dataframe(by_sym_display.reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    with st.expander("Full prediction log"):
        log = hist[["symbol", "name", "predicted_for", "predicted_direction",
                    "actual_direction", "correct", "confidence",
                    "predicted_close", "actual_close"]].copy()
        log["confidence"]      = log["confidence"].apply(lambda x: f"{x*100:.1f}%")
        log["correct"]         = log["correct"].apply(lambda x: "✅" if x else "❌")
        log["predicted_close"] = log["predicted_close"].apply(lambda x: f"{x:.2f}")
        log["actual_close"]    = log["actual_close"].apply(lambda x: f"{x:.2f}")
        log.columns = [
            "Symbol", "Name", "Date", "Predicted", "Actual",
            "Result", "Confidence", "Predicted Close", "Actual Close",
        ]
        st.dataframe(log.reset_index(drop=True), use_container_width=True)


def page_prediction_vs_actual():
    st.title("Predicted vs Actual — Close Price")

    securities = load_securities()
    if securities.empty:
        st.warning("No securities loaded yet. Go to **Setup / Refresh** first.")
        return

    col_sel, col_hor = st.columns([3, 1])
    with col_sel:
        securities["label"] = securities["symbol"] + " — " + securities["name"]
        label_to_symbol = dict(zip(securities["label"], securities["symbol"]))
        selected_label = st.selectbox("Select listing", securities["label"].tolist(), key="pva_symbol")
        symbol = label_to_symbol[selected_label]
    with col_hor:
        horizon = st.select_slider(
            "Horizon",
            options=list(range(1, 8)),
            value=1,
            format_func=lambda n: f"{n}d",
            key="pva_horizon",
        )

    hist = load_symbol_prediction_history(symbol, horizon)

    if hist.empty:
        st.info(
            "No past predictions found for this listing. "
            "History appears once predictions have been made and the target date has passed."
        )
        return

    hist = hist.sort_values("predicted_for")

    # ── Summary metrics ──────────────────────────────────────────────────────
    total     = len(hist)
    dir_acc   = hist["correct"].mean() * 100
    mae       = (hist["predicted_close"] - hist["actual_close"]).abs().mean()
    mape      = ((hist["predicted_close"] - hist["actual_close"]).abs() / hist["actual_close"].replace(0, float("nan"))).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days Evaluated",       f"{total}")
    c2.metric("Direction Accuracy",   f"{dir_acc:.1f}%")
    c3.metric("Avg Price Error (MAE)", f"LKR {mae:.2f}")
    c4.metric("Avg Price Error (MAPE)", f"{mape:.1f}%")

    st.markdown("---")

    # ── Predicted vs Actual close chart ──────────────────────────────────────
    correct_mask   = hist["correct"] == 1
    incorrect_mask = hist["correct"] == 0

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist["predicted_for"], y=hist["actual_close"],
        mode="lines+markers",
        name="Actual Close",
        line=dict(color="#90caf9", width=2),
        marker=dict(size=5),
    ))

    fig.add_trace(go.Scatter(
        x=hist["predicted_for"], y=hist["predicted_close"],
        mode="lines+markers",
        name="Predicted Close",
        line=dict(color="#ffb74d", width=2, dash="dash"),
        marker=dict(size=5),
    ))

    # Overlay correct/incorrect direction markers on the actual line
    if correct_mask.any():
        fig.add_trace(go.Scatter(
            x=hist.loc[correct_mask, "predicted_for"],
            y=hist.loc[correct_mask, "actual_close"],
            mode="markers",
            name="Direction Correct ✅",
            marker=dict(color="#26a69a", size=9, symbol="circle"),
            hovertemplate="%{x}<br>Actual: %{y:.2f}<br>Direction: correct<extra></extra>",
        ))

    if incorrect_mask.any():
        fig.add_trace(go.Scatter(
            x=hist.loc[incorrect_mask, "predicted_for"],
            y=hist.loc[incorrect_mask, "actual_close"],
            mode="markers",
            name="Direction Wrong ❌",
            marker=dict(color="#ef5350", size=9, symbol="x"),
            hovertemplate="%{x}<br>Actual: %{y:.2f}<br>Direction: wrong<extra></extra>",
        ))

    fig.update_layout(
        title=f"{symbol} — Predicted vs Actual Close ({horizon}-day horizon)",
        xaxis_title="Date",
        yaxis_title="Price (LKR)",
        template="plotly_dark",
        height=440,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Price error over time ─────────────────────────────────────────────────
    hist_err = hist.copy()
    hist_err["price_error"] = hist_err["predicted_close"] - hist_err["actual_close"]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=hist_err["predicted_for"],
        y=hist_err["price_error"],
        name="Prediction Error",
        marker_color=[
            "#26a69a" if e >= 0 else "#ef5350" for e in hist_err["price_error"]
        ],
        hovertemplate="%{x}<br>Error: %{y:+.2f} LKR<extra></extra>",
    ))
    fig2.add_hline(y=0, line_dash="dot", line_color="#888")
    fig2.update_layout(
        title="Daily Price Prediction Error (Predicted − Actual)",
        xaxis_title="Date",
        yaxis_title="Error (LKR)",
        template="plotly_dark",
        height=280,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Raw data table ────────────────────────────────────────────────────────
    with st.expander("Raw data"):
        log = hist[["predicted_for", "predicted_direction", "actual_direction",
                    "correct", "confidence", "predicted_close", "actual_close"]].copy()
        log["confidence"]      = log["confidence"].apply(lambda x: f"{x*100:.1f}%")
        log["correct"]         = log["correct"].apply(lambda x: "✅" if x else "❌")
        log["predicted_close"] = log["predicted_close"].apply(lambda x: f"{x:.2f}")
        log["actual_close"]    = log["actual_close"].apply(lambda x: f"{x:.2f}")
        log = log.sort_values("predicted_for", ascending=False)
        log.columns = ["Date", "Predicted Dir", "Actual Dir", "Result",
                       "Confidence", "Predicted Close", "Actual Close"]
        st.dataframe(log.reset_index(drop=True), use_container_width=True)


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
        retrain_horizon = st.select_slider(
            "Horizon to retrain",
            options=list(range(1, 8)),
            value=1,
            format_func=lambda n: f"{n} day" if n == 1 else f"{n} days",
            key="setup_retrain_horizon",
        )
        if st.button("🤖 Retrain all models + predict"):
            with st.spinner(f"Training {retrain_horizon}-day models …"):
                predictor.train_all(retrain_horizon)
                predictor.predict_all(retrain_horizon)
            st.success(f"All {retrain_horizon}-day models retrained and predictions saved.")
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
elif page == "Predicted vs Actual":
    page_prediction_vs_actual()
elif page == "Setup / Refresh":
    page_setup()
elif page == "Admin Panel":
    page_admin()

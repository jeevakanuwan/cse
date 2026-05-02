"""
Google OAuth 2.0 authentication for the CSE dashboard.

Flow:
  1. User clicks "Sign in with Google" → redirected to Google consent screen
  2. Google redirects back to app with ?code=...&state=...
  3. handle_oauth_callback() exchanges code for access token
  4. User info fetched from Google; role resolved from DB
  5. User info + role stored in st.session_state for the session
"""

import hmac
import hashlib
import secrets
import requests
import streamlit as st
from urllib.parse import urlencode

from . import database as db

GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_INFO_URL  = "https://www.googleapis.com/oauth2/v3/userinfo"


def _cfg():
    try:
        import config
        return config
    except ImportError:
        return None


def _sign_state(token: str) -> str:
    """Return HMAC-signed state — verifiable without server-side session storage."""
    cfg = _cfg()
    secret = getattr(cfg, "GOOGLE_CLIENT_SECRET", "fallback-secret").encode()
    sig = hmac.new(secret, token.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{token}.{sig}"


def _verify_state(state: str) -> bool:
    if not state or "." not in state:
        return False
    token, _ = state.rsplit(".", 1)
    return hmac.compare_digest(_sign_state(token), state)


def google_login_url() -> str:
    """Generate the Google OAuth consent screen URL."""
    cfg = _cfg()
    state = _sign_state(secrets.token_urlsafe(16))
    redirect_uri = cfg.REDIRECT_URI.rstrip("/") + "/"
    params = {
        "client_id":     cfg.GOOGLE_CLIENT_ID,
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "scope":         "openid email profile",
        "state":         state,
        "access_type":   "online",
        "prompt":        "select_account",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def handle_oauth_callback() -> bool:
    """
    Detect ?code=... in URL after Google redirect, exchange it for a token,
    fetch user info, and store the session. Returns True if a callback was handled.
    """
    params = st.query_params
    if "code" not in params:
        return False

    cfg = _cfg()
    code  = params.get("code", "")
    state = params.get("state", "")

    # CSRF check — HMAC-based so it survives the browser redirect
    # (st.session_state is lost when the user is redirected to Google and back)
    if not _verify_state(state):
        st.error("Login failed: invalid state. Please try again.")
        st.query_params.clear()
        return True

    redirect_uri = cfg.REDIRECT_URI.rstrip("/") + "/"

    # Exchange code for access token
    try:
        token_resp = requests.post(GOOGLE_TOKEN_URL, data={
            "code":          code,
            "client_id":     cfg.GOOGLE_CLIENT_ID,
            "client_secret": cfg.GOOGLE_CLIENT_SECRET,
            "redirect_uri":  redirect_uri,
            "grant_type":    "authorization_code",
        }, timeout=15)
        token_data = token_resp.json()
    except Exception as exc:
        st.error(f"Login failed: could not contact Google. {exc}")
        st.query_params.clear()
        return True

    access_token = token_data.get("access_token")
    if not access_token:
        st.error(f"Login failed: {token_data.get('error_description', 'no token received')}")
        st.query_params.clear()
        return True

    # Fetch user profile
    try:
        info = requests.get(
            GOOGLE_INFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        ).json()
    except Exception as exc:
        st.error(f"Login failed: could not fetch Google profile. {exc}")
        st.query_params.clear()
        return True

    email   = info.get("email", "")
    name    = info.get("name", email)
    picture = info.get("picture", "")

    if not info.get("email_verified", False):
        st.error("Login failed: Google account email is not verified.")
        st.query_params.clear()
        return True

    # Upsert user in DB and get role
    role = db.upsert_user(email, name, picture)
    st.session_state["user"] = {"email": email, "name": name, "picture": picture}
    st.session_state["role"] = role
    st.query_params.clear()
    return True


def get_user() -> dict | None:
    return st.session_state.get("user")


def get_role() -> str | None:
    return st.session_state.get("role")


def logout():
    for key in ("user", "role"):
        st.session_state.pop(key, None)
    st.query_params.clear()


def is_configured() -> bool:
    """Returns False if Google OAuth credentials are not yet set in config.py."""
    cfg = _cfg()
    if cfg is None:
        return False
    client_id     = getattr(cfg, "GOOGLE_CLIENT_ID", "")
    client_secret = getattr(cfg, "GOOGLE_CLIENT_SECRET", "")
    return (
        client_id.endswith(".apps.googleusercontent.com") and
        client_secret not in ("", "your-client-secret")
    )

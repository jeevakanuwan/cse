"""
Tests for src/auth.py

Covers: HMAC state signing and verification, timing-safe comparison,
is_configured() logic, and google_login_url() URL construction.
No real HTTP requests are made; st (Streamlit) is mocked throughout.
"""

import sys
import hmac
import pytest
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse, parse_qs


# ===========================================================================
# State signing and verification
# ===========================================================================

class TestSignState:
    def test_output_contains_exactly_one_dot(self, mock_config):
        from src.auth import _sign_state
        state = _sign_state("sometoken")
        assert state.count(".") == 1

    def test_token_part_unchanged(self, mock_config):
        from src.auth import _sign_state
        state = _sign_state("mytoken")
        token, _sig = state.rsplit(".", 1)
        assert token == "mytoken"

    def test_signature_is_16_hex_characters(self, mock_config):
        from src.auth import _sign_state
        state = _sign_state("mytoken")
        _token, sig = state.rsplit(".", 1)
        assert len(sig) == 16
        assert all(c in "0123456789abcdef" for c in sig)

    def test_deterministic_for_same_input(self, mock_config):
        from src.auth import _sign_state
        assert _sign_state("abc") == _sign_state("abc")

    def test_different_tokens_different_signatures(self, mock_config):
        from src.auth import _sign_state
        s1 = _sign_state("token-one")
        s2 = _sign_state("token-two")
        assert s1 != s2

    def test_different_secrets_produce_different_signatures(self):
        import src.auth as auth
        cfg1, cfg2 = MagicMock(), MagicMock()
        cfg1.GOOGLE_CLIENT_SECRET = "secret-alpha"
        cfg2.GOOGLE_CLIENT_SECRET = "secret-beta"
        with patch.object(auth, "_cfg", return_value=cfg1):
            s1 = auth._sign_state("token")
        with patch.object(auth, "_cfg", return_value=cfg2):
            s2 = auth._sign_state("token")
        assert s1 != s2


class TestVerifyState:
    def test_valid_own_signature_returns_true(self, mock_config):
        from src.auth import _sign_state, _verify_state
        state = _sign_state("validtoken")
        assert _verify_state(state) is True

    def test_empty_string_returns_false(self, mock_config):
        from src.auth import _verify_state
        assert _verify_state("") is False

    def test_no_dot_separator_returns_false(self, mock_config):
        from src.auth import _verify_state
        assert _verify_state("nodothere") is False

    def test_tampered_token_returns_false(self, mock_config):
        from src.auth import _sign_state, _verify_state
        state   = _sign_state("original")
        _sig    = state.rsplit(".", 1)[1]
        tampered = f"tampered.{_sig}"
        assert _verify_state(tampered) is False

    def test_tampered_signature_returns_false(self, mock_config):
        from src.auth import _sign_state, _verify_state
        state  = _sign_state("original")
        token  = state.rsplit(".", 1)[0]
        tampered = f"{token}.0000000000000000"
        assert _verify_state(tampered) is False

    def test_signature_one_char_off_returns_false(self, mock_config):
        from src.auth import _sign_state, _verify_state
        state = _sign_state("original")
        token, sig = state.rsplit(".", 1)
        # Flip the last character of the signature
        flipped = sig[:-1] + ("a" if sig[-1] != "a" else "b")
        assert _verify_state(f"{token}.{flipped}") is False

    def test_uses_hmac_compare_digest(self, mock_config):
        """Comparison must be timing-safe (constant-time)."""
        import src.auth as auth
        from src.auth import _sign_state
        state = _sign_state("sometoken")
        with patch.object(hmac, "compare_digest",
                          wraps=hmac.compare_digest) as mock_cd:
            auth._verify_state(state)
        mock_cd.assert_called_once()

    def test_any_non_empty_valid_token_verifies(self, mock_config):
        from src.auth import _sign_state, _verify_state
        for token in ["abc", "x" * 50, "123-ABC_xyz"]:
            assert _verify_state(_sign_state(token)) is True


# ===========================================================================
# is_configured
# ===========================================================================

class TestIsConfigured:
    def test_returns_true_when_fully_configured(self, mock_config):
        from src.auth import is_configured
        mock_config.GOOGLE_CLIENT_ID     = "123.apps.googleusercontent.com"
        mock_config.GOOGLE_CLIENT_SECRET = "GOCSPX-real-secret-abc"
        assert is_configured() is True

    def test_returns_false_with_placeholder_secret(self, mock_config):
        from src.auth import is_configured
        mock_config.GOOGLE_CLIENT_ID     = "123.apps.googleusercontent.com"
        mock_config.GOOGLE_CLIENT_SECRET = "your-client-secret"
        assert is_configured() is False

    def test_returns_false_with_empty_secret(self, mock_config):
        from src.auth import is_configured
        mock_config.GOOGLE_CLIENT_ID     = "123.apps.googleusercontent.com"
        mock_config.GOOGLE_CLIENT_SECRET = ""
        assert is_configured() is False

    def test_returns_false_when_client_id_missing_google_suffix(self, mock_config):
        from src.auth import is_configured
        mock_config.GOOGLE_CLIENT_ID     = "not-a-google-client-id"
        mock_config.GOOGLE_CLIENT_SECRET = "GOCSPX-real-secret"
        assert is_configured() is False

    def test_returns_false_when_config_import_fails(self):
        sys.modules.pop("config", None)
        import src.auth as auth
        with patch.object(auth, "_cfg", return_value=None):
            assert auth.is_configured() is False

    def test_empty_client_id_returns_false(self, mock_config):
        from src.auth import is_configured
        mock_config.GOOGLE_CLIENT_ID     = ""
        mock_config.GOOGLE_CLIENT_SECRET = "GOCSPX-real-secret"
        assert is_configured() is False


# ===========================================================================
# google_login_url
# ===========================================================================

class TestGoogleLoginUrl:
    def _get_url(self, mock_config):
        import src.auth as auth
        with patch("src.auth.st") as mock_st:
            mock_st.session_state = {}
            return auth.google_login_url()

    def test_points_to_google_oauth_endpoint(self, mock_config):
        url = self._get_url(mock_config)
        assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth")

    def test_contains_client_id(self, mock_config):
        url = self._get_url(mock_config)
        params = parse_qs(urlparse(url).query)
        assert params["client_id"][0] == mock_config.GOOGLE_CLIENT_ID

    def test_response_type_is_code(self, mock_config):
        url = self._get_url(mock_config)
        params = parse_qs(urlparse(url).query)
        assert params["response_type"][0] == "code"

    def test_scope_includes_openid_email_profile(self, mock_config):
        url = self._get_url(mock_config)
        params = parse_qs(urlparse(url).query)
        scope = params["scope"][0]
        assert "openid" in scope
        assert "email"  in scope
        assert "profile" in scope

    def test_state_verifies_with_own_signature(self, mock_config):
        import src.auth as auth
        url = self._get_url(mock_config)
        params = parse_qs(urlparse(url).query)
        state = params["state"][0]
        assert auth._verify_state(state) is True

    def test_redirect_uri_has_trailing_slash(self, mock_config):
        mock_config.REDIRECT_URI = "http://example.com"   # deliberately no slash
        url = self._get_url(mock_config)
        params = parse_qs(urlparse(url).query)
        assert params["redirect_uri"][0].endswith("/")

    def test_redirect_uri_not_double_slashed(self, mock_config):
        mock_config.REDIRECT_URI = "http://example.com/"  # already has trailing slash
        url = self._get_url(mock_config)
        params = parse_qs(urlparse(url).query)
        assert not params["redirect_uri"][0].endswith("//")

    def test_each_call_produces_unique_state(self, mock_config):
        import src.auth as auth
        states = set()
        with patch("src.auth.st") as mock_st:
            mock_st.session_state = {}
            for _ in range(10):
                url   = auth.google_login_url()
                state = parse_qs(urlparse(url).query)["state"][0]
                states.add(state)
        assert len(states) == 10   # all unique


# ===========================================================================
# handle_oauth_callback  (unit-level checks)
# ===========================================================================

class TestHandleOAuthCallback:
    def _make_st_mock(self, params: dict):
        mock_st = MagicMock()
        mock_st.query_params = params
        mock_st.session_state = {}
        return mock_st

    def test_returns_false_when_no_code_in_params(self, mock_config):
        import src.auth as auth
        with patch("src.auth.st") as mock_st:
            mock_st.query_params = {}
            result = auth.handle_oauth_callback()
        assert result is False

    def test_returns_true_and_shows_error_for_invalid_state(self, mock_config):
        import src.auth as auth
        with patch("src.auth.st") as mock_st:
            mock_st.query_params = {"code": "abc123", "state": "invalid.state000"}
            mock_st.session_state = {}
            result = auth.handle_oauth_callback()
        assert result is True
        mock_st.error.assert_called_once()

    def test_shows_error_on_token_exchange_failure(self, mock_config):
        import src.auth as auth
        from src.auth import _sign_state
        state = _sign_state("validtoken")
        mock_token_resp = MagicMock()
        mock_token_resp.json.return_value = {
            "error": "invalid_grant",
            "error_description": "Code already used",
        }
        with patch("src.auth.st") as mock_st, \
             patch("src.auth.requests.post", return_value=mock_token_resp):
            mock_st.query_params = {"code": "expired-code", "state": state}
            mock_st.session_state = {}
            result = auth.handle_oauth_callback()
        assert result is True
        mock_st.error.assert_called_once()

    def test_successful_callback_stores_user_in_session(self, mock_config, tmp_db):
        import src.auth as auth
        from src.auth import _sign_state
        state = _sign_state("validtoken")
        mock_config.ADMIN_EMAILS = []

        token_resp   = MagicMock()
        token_resp.json.return_value = {"access_token": "google-access-token"}

        userinfo_resp = MagicMock()
        userinfo_resp.json.return_value = {
            "email":          "newuser@example.com",
            "name":           "New User",
            "picture":        "https://example.com/pic.jpg",
            "email_verified": True,
        }

        session_state = {}
        with patch("src.auth.st") as mock_st, \
             patch("src.auth.requests.post", return_value=token_resp), \
             patch("src.auth.requests.get",  return_value=userinfo_resp), \
             patch("src.auth.db",            tmp_db):
            mock_st.query_params = {"code": "valid-code", "state": state}
            mock_st.session_state = session_state
            result = auth.handle_oauth_callback()

        assert result is True
        assert session_state["user"]["email"] == "newuser@example.com"
        assert session_state["user"]["name"]  == "New User"
        assert session_state["role"] == "pending"

    def test_unverified_email_rejected(self, mock_config):
        import src.auth as auth
        from src.auth import _sign_state
        state = _sign_state("validtoken")

        token_resp    = MagicMock()
        token_resp.json.return_value = {"access_token": "google-access-token"}
        userinfo_resp = MagicMock()
        userinfo_resp.json.return_value = {
            "email": "unverified@example.com", "name": "User",
            "picture": "", "email_verified": False,
        }

        with patch("src.auth.st") as mock_st, \
             patch("src.auth.requests.post", return_value=token_resp), \
             patch("src.auth.requests.get",  return_value=userinfo_resp):
            mock_st.query_params = {"code": "code", "state": state}
            mock_st.session_state = {}
            result = auth.handle_oauth_callback()

        assert result is True
        mock_st.error.assert_called_once()
        assert "email" not in mock_st.session_state

"""
Tests for src/scraper.py

Covers: date utilities, safe type conversions, HTTP response parsing for
fetch_history / fetch_securities / fetch_daily_summary, and login logic.
All network calls are mocked — no real HTTP requests are made.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixture: reset module-level _logged_in between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_logged_in():
    import src.scraper as scraper
    scraper._logged_in = False
    yield
    scraper._logged_in = False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _mock_response(data, status_code=200):
    resp = MagicMock()
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    resp.status_code = status_code
    return resp


# ===========================================================================
# Date utilities
# ===========================================================================

class TestToCseDate:
    def test_standard_date(self):
        from src.scraper import _to_cse_date
        assert _to_cse_date("2024-03-15") == "15-03-2024"

    def test_january_first(self):
        from src.scraper import _to_cse_date
        assert _to_cse_date("2024-01-01") == "01-01-2024"

    def test_leap_day(self):
        from src.scraper import _to_cse_date
        assert _to_cse_date("2024-02-29") == "29-02-2024"

    def test_year_end(self):
        from src.scraper import _to_cse_date
        assert _to_cse_date("2023-12-31") == "31-12-2023"


class TestParseCseDate:
    def test_dd_mm_yyyy(self):
        from src.scraper import _parse_cse_date
        assert _parse_cse_date("15-03-2024") == "2024-03-15"

    def test_yyyy_mm_dd(self):
        from src.scraper import _parse_cse_date
        assert _parse_cse_date("2024-03-15") == "2024-03-15"

    def test_dd_slash_mm_slash_yyyy(self):
        from src.scraper import _parse_cse_date
        assert _parse_cse_date("15/03/2024") == "2024-03-15"

    def test_yyyy_slash_mm_slash_dd(self):
        from src.scraper import _parse_cse_date
        assert _parse_cse_date("2024/03/15") == "2024-03-15"

    def test_invalid_returns_none(self):
        from src.scraper import _parse_cse_date
        assert _parse_cse_date("not-a-date") is None

    def test_empty_string_returns_none(self):
        from src.scraper import _parse_cse_date
        assert _parse_cse_date("") is None

    def test_strips_surrounding_whitespace(self):
        from src.scraper import _parse_cse_date
        assert _parse_cse_date("  15-03-2024  ") == "2024-03-15"


# ===========================================================================
# Safe type conversions
# ===========================================================================

class TestSafeFloat:
    def test_valid_float(self):
        from src.scraper import _safe_float
        assert _safe_float({"p": 1.5}, "p") == pytest.approx(1.5)

    def test_valid_string_number(self):
        from src.scraper import _safe_float
        assert _safe_float({"p": "3.14"}, "p") == pytest.approx(3.14)

    def test_integer_value(self):
        from src.scraper import _safe_float
        assert _safe_float({"p": 42}, "p") == pytest.approx(42.0)

    def test_none_value_returns_none(self):
        from src.scraper import _safe_float
        assert _safe_float({"p": None}, "p") is None

    def test_empty_string_returns_none(self):
        from src.scraper import _safe_float
        assert _safe_float({"p": ""}, "p") is None

    def test_na_string_returns_none(self):
        from src.scraper import _safe_float
        assert _safe_float({"p": "N/A"}, "p") is None

    def test_missing_key_returns_none(self):
        from src.scraper import _safe_float
        assert _safe_float({}, "p") is None

    def test_unparseable_string_returns_none(self):
        from src.scraper import _safe_float
        assert _safe_float({"p": "not-a-number"}, "p") is None


class TestSafeInt:
    def test_valid_int(self):
        from src.scraper import _safe_int
        assert _safe_int({"v": 42}, "v") == 42

    def test_float_string_truncated(self):
        from src.scraper import _safe_int
        assert _safe_int({"v": "1000.0"}, "v") == 1000

    def test_none_returns_zero(self):
        from src.scraper import _safe_int
        assert _safe_int({"v": None}, "v") == 0

    def test_empty_string_returns_zero(self):
        from src.scraper import _safe_int
        assert _safe_int({"v": ""}, "v") == 0

    def test_na_string_returns_zero(self):
        from src.scraper import _safe_int
        assert _safe_int({"v": "N/A"}, "v") == 0

    def test_missing_key_returns_zero(self):
        from src.scraper import _safe_int
        assert _safe_int({}, "v") == 0

    def test_large_volume(self):
        from src.scraper import _safe_int
        assert _safe_int({"v": 9_876_543}, "v") == 9_876_543


# ===========================================================================
# fetch_history  (HTTP mocked)
# ===========================================================================

class TestFetchHistory:
    def _ts(self, date_str):
        """Millisecond timestamp for a given YYYY-MM-DD (UTC midnight)."""
        from datetime import timezone
        y, m, d = (int(x) for x in date_str.split("-"))
        return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)

    def test_parses_millisecond_trade_date(self, mock_config):
        import src.scraper as scraper
        resp_data = [{"tradeDate": self._ts("2024-03-15"),
                      "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5,
                      "shareVolume": 5000, "tradeVolume": 100}]
        with patch.object(scraper.SESSION, "post",
                          return_value=_mock_response(resp_data)):
            records = scraper.fetch_history("TEST.N0000", "2024-03-01", "2024-03-31")
        assert len(records) == 1
        assert records[0]["date"]   == "2024-03-15"
        assert records[0]["close"]  == pytest.approx(10.5)
        assert records[0]["symbol"] == "TEST.N0000"

    def test_uses_share_volume_and_trade_volume(self, mock_config):
        import src.scraper as scraper
        resp_data = [{"tradeDate": self._ts("2024-03-15"),
                      "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5,
                      "shareVolume": 9876, "tradeVolume": 200}]
        with patch.object(scraper.SESSION, "post",
                          return_value=_mock_response(resp_data)):
            records = scraper.fetch_history("TEST.N0000", "2024-03-01", "2024-03-31")
        assert records[0]["volume"] == 9876
        assert records[0]["trades"] == 200

    def test_null_open_and_low_preserved(self, mock_config):
        import src.scraper as scraper
        resp_data = [{"tradeDate": self._ts("2024-03-15"),
                      "open": None, "high": 11.0, "low": None, "close": 10.5,
                      "shareVolume": 500, "tradeVolume": 50}]
        with patch.object(scraper.SESSION, "post",
                          return_value=_mock_response(resp_data)):
            records = scraper.fetch_history("TEST.N0000", "2024-03-01", "2024-03-31")
        assert records[0]["open"] is None
        assert records[0]["low"]  is None

    def test_empty_response_returns_empty_list(self, mock_config):
        import src.scraper as scraper
        with patch.object(scraper.SESSION, "post",
                          return_value=_mock_response([])):
            records = scraper.fetch_history("TEST.N0000", "2024-03-01", "2024-03-31")
        assert records == []

    def test_response_wrapped_in_data_key(self, mock_config):
        import src.scraper as scraper
        resp_data = {"data": [{"tradeDate": self._ts("2024-03-15"),
                               "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5,
                               "shareVolume": 100, "tradeVolume": 5}]}
        with patch.object(scraper.SESSION, "post",
                          return_value=_mock_response(resp_data)):
            records = scraper.fetch_history("TEST.N0000", "2024-03-01", "2024-03-31")
        assert len(records) == 1

    def test_skips_rows_with_missing_trade_date(self, mock_config):
        import src.scraper as scraper
        resp_data = [
            {"tradeDate": None, "open": 10.0, "high": 11.0,
             "low": 9.5, "close": 10.5, "shareVolume": 100, "tradeVolume": 5},
            {"tradeDate": self._ts("2024-03-15"), "open": 10.0, "high": 11.0,
             "low": 9.5, "close": 10.5, "shareVolume": 200, "tradeVolume": 10},
        ]
        with patch.object(scraper.SESSION, "post",
                          return_value=_mock_response(resp_data)):
            records = scraper.fetch_history("TEST.N0000", "2024-03-01", "2024-03-31")
        assert len(records) == 1

    def test_sends_date_in_dd_mm_yyyy_format(self, mock_config):
        import src.scraper as scraper
        captured = {}
        def fake_post(url, data=None, headers=None, **kwargs):
            captured["data"] = data
            return _mock_response([])
        with patch.object(scraper.SESSION, "post", side_effect=fake_post):
            scraper.fetch_history("ABAN.N0000", "2024-01-15", "2024-03-20")
        assert captured["data"]["fromDate"] == "15-01-2024"
        assert captured["data"]["toDate"]   == "20-03-2024"
        assert captured["data"]["symbol"]   == "ABAN.N0000"

    def test_multiple_rows_all_parsed(self, mock_config):
        import src.scraper as scraper
        resp_data = [
            {"tradeDate": self._ts(f"2024-03-{i:02d}"), "open": 10.0, "high": 11.0,
             "low": 9.5, "close": 10.0 + i * 0.1, "shareVolume": 100, "tradeVolume": 5}
            for i in range(1, 6)
        ]
        with patch.object(scraper.SESSION, "post",
                          return_value=_mock_response(resp_data)):
            records = scraper.fetch_history("TEST.N0000", "2024-03-01", "2024-03-31")
        assert len(records) == 5

    def test_zero_volume_stored_as_zero(self, mock_config):
        import src.scraper as scraper
        resp_data = [{"tradeDate": self._ts("2024-03-15"),
                      "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5,
                      "shareVolume": None, "tradeVolume": None}]
        with patch.object(scraper.SESSION, "post",
                          return_value=_mock_response(resp_data)):
            records = scraper.fetch_history("TEST.N0000", "2024-03-01", "2024-03-31")
        assert records[0]["volume"] == 0
        assert records[0]["trades"] == 0


# ===========================================================================
# fetch_securities  (HTTP mocked)
# ===========================================================================

class TestFetchSecurities:
    def test_parses_symbol_and_name(self, mock_config):
        import src.scraper as scraper
        sec_resp = [
            {"symbol": "JKH.N0000",  "name": "John Keells Holdings PLC"},
            {"symbol": "DIAL.N0000", "name": "Dialog Axiata PLC"},
        ]
        with patch.object(scraper.SESSION, "get",
                          return_value=_mock_response(sec_resp)), \
             patch.object(scraper.SESSION, "post",
                          return_value=_mock_response([])):
            result = scraper.fetch_securities()
        assert len(result) == 2
        symbols = {s["symbol"] for s in result}
        assert "JKH.N0000"  in symbols
        assert "DIAL.N0000" in symbols

    def test_assigns_unknown_sector_when_no_mapping(self, mock_config):
        import src.scraper as scraper
        sec_resp = [{"symbol": "JKH.N0000", "name": "John Keells"}]
        with patch.object(scraper.SESSION, "get",
                          return_value=_mock_response(sec_resp)), \
             patch.object(scraper.SESSION, "post",
                          return_value=_mock_response([])):
            result = scraper.fetch_securities()
        assert result[0]["sector"] == "Unknown"

    def test_empty_response_returns_empty_list(self, mock_config):
        import src.scraper as scraper
        with patch.object(scraper.SESSION, "get",
                          return_value=_mock_response([])), \
             patch.object(scraper.SESSION, "post",
                          return_value=_mock_response([])):
            result = scraper.fetch_securities()
        assert result == []

    def test_skips_entries_without_symbol(self, mock_config):
        import src.scraper as scraper
        sec_resp = [
            {"symbol": "JKH.N0000", "name": "JKH"},
            {"name": "No Symbol Here"},    # no symbol key
            {"symbol": "",  "name": "Empty Symbol"},
        ]
        with patch.object(scraper.SESSION, "get",
                          return_value=_mock_response(sec_resp)), \
             patch.object(scraper.SESSION, "post",
                          return_value=_mock_response([])):
            result = scraper.fetch_securities()
        assert all(r["symbol"] for r in result)
        assert len(result) == 1


# ===========================================================================
# Login
# ===========================================================================

class TestLogin:
    def test_uses_jwt_token_from_config(self, mock_config):
        import src.scraper as scraper
        mock_config.CSE_TOKEN = "eyJtest.valid.token"
        result = scraper.login()
        assert result is True
        assert scraper._logged_in is True

    def test_strips_extra_ga_cookies_after_semicolon(self, mock_config):
        import src.scraper as scraper
        # Simulate user pasting full cookie string
        mock_config.CSE_TOKEN = (
            "eyJtoken.only.this; _ga=blah; _ga_OTHER=blah"
        )
        result = scraper.login()
        assert result is True
        # Verify only the JWT part was set as the cookie
        cookie = scraper.SESSION.cookies.get("accessToken", domain="www.cse.lk")
        assert ";" not in (cookie or "")

    def test_returns_false_when_token_and_ropc_both_fail(self, mock_config):
        import src.scraper as scraper
        mock_config.CSE_TOKEN  = ""
        mock_config.CSE_EMAIL  = "user@test.com"
        mock_config.CSE_PASSWORD = "pass"
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {}
        with patch("requests.post", return_value=mock_resp):
            result = scraper.login()
        assert result is False

    def test_returns_false_when_config_missing(self):
        import sys, src.scraper as scraper
        with patch.dict(sys.modules, {"config": None}):
            result = scraper.login()
        assert result is False


# ===========================================================================
# Build sector map edge cases
# ===========================================================================

class TestBuildSectorMap:
    def test_nested_securities_format(self, mock_config):
        """Sector data in the form [{sector: 'X', securities: [{symbol: ...}]}]."""
        import src.scraper as scraper
        sectors_resp = [
            {"sector": "Banking", "securities": [
                {"symbol": "COMB.N0000"},
                {"symbol": "HNB.N0000"},
            ]},
        ]
        with patch.object(scraper.SESSION, "post",
                          return_value=_mock_response(sectors_resp)):
            mapping = scraper._build_sector_map()
        assert mapping.get("COMB.N0000") == "Banking"
        assert mapping.get("HNB.N0000")  == "Banking"

    def test_flat_format(self, mock_config):
        """Sector data where each row is {symbol: ..., sector: ...}."""
        import src.scraper as scraper
        sectors_resp = [
            {"symbol": "JKH.N0000",  "sector": "Diversified"},
            {"symbol": "DIAL.N0000", "sector": "Telecom"},
        ]
        with patch.object(scraper.SESSION, "post",
                          return_value=_mock_response(sectors_resp)):
            mapping = scraper._build_sector_map()
        assert mapping.get("JKH.N0000")  == "Diversified"
        assert mapping.get("DIAL.N0000") == "Telecom"

    def test_empty_response_returns_empty_dict(self, mock_config):
        import src.scraper as scraper
        with patch.object(scraper.SESSION, "post",
                          return_value=_mock_response([])):
            mapping = scraper._build_sector_map()
        assert mapping == {}

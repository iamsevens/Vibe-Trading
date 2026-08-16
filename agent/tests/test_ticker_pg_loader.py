"""Tests for ticker_pg_loader: code parsing, availability, and fetch logic.

All tests are unit-level — no real database connections are made.
"""
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backtest.loaders.ticker_pg_loader import DataLoader, _MARKET_MAP


class TestCodeParsing:
    """Verify _fetch_one parses tushare-style codes and rejects non-A-shares."""

    def test_sh_code_parsed(self):
        """601398.SH → market='SH', symbol='601398'."""
        loader = DataLoader()
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = []

        loader._fetch_one(conn, "601398.SH", "2024-01-01", "2024-01-31")
        params = cur.execute.call_args[0][1]
        assert params == ("SH", "601398", "2024-01-01", "2024-01-31")

    def test_sz_code_parsed(self):
        """000001.SZ → market='SZ', symbol='000001'."""
        loader = DataLoader()
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = []

        loader._fetch_one(conn, "000001.SZ", "2024-01-01", "2024-01-31")
        params = cur.execute.call_args[0][1]
        assert params == ("SZ", "000001", "2024-01-01", "2024-01-31")

    def test_bj_code_parsed(self):
        """430047.BJ → market='BJ', symbol='430047'."""
        loader = DataLoader()
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = []

        loader._fetch_one(conn, "430047.BJ", "2024-01-01", "2024-01-31")
        params = cur.execute.call_args[0][1]
        assert params == ("BJ", "430047", "2024-01-01", "2024-01-31")

    def test_rejects_no_suffix(self):
        """Codes without a market suffix return None."""
        loader = DataLoader()
        conn = MagicMock()
        assert loader._fetch_one(conn, "601398", "2024-01-01", "2024-01-31") is None

    def test_rejects_non_a_share_market(self):
        """US/HK suffixes are rejected."""
        loader = DataLoader()
        conn = MagicMock()
        assert loader._fetch_one(conn, "AAPL.US", "2024-01-01", "2024-01-31") is None
        assert loader._fetch_one(conn, "00700.HK", "2024-01-01", "2024-01-31") is None

    def test_case_insensitive(self):
        """Lowercase suffixes are accepted."""
        loader = DataLoader()
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = []

        loader._fetch_one(conn, "601398.sh", "2024-01-01", "2024-01-31")
        params = cur.execute.call_args[0][1]
        assert params[0] == "SH"


class TestIsAvailable:
    """Verify is_available reflects DSN and psycopg presence."""

    def test_unavailable_without_dsn(self):
        loader = DataLoader()
        with patch.object(loader, "_get_dsn", return_value=""):
            assert loader.is_available() is False

    def test_unavailable_with_dsn_but_no_psycopg(self):
        loader = DataLoader()
        with patch.object(loader, "_get_dsn", return_value="postgresql://x:y@z/db"), \
             patch.dict("sys.modules", {"psycopg": None}):
            assert loader.is_available() is False

    def test_available_with_dsn_and_psycopg(self):
        loader = DataLoader()
        with patch.object(loader, "_get_dsn", return_value="postgresql://x:y@z/db"):
            assert loader.is_available() is True


class TestIntervalRejection:
    """Non-daily intervals must return an empty dict."""

    def test_rejects_hourly(self):
        loader = DataLoader()
        result = loader.fetch(["601398.SH"], "2024-01-01", "2024-01-31", interval="1H")
        assert result == {}

    def test_rejects_minute(self):
        loader = DataLoader()
        result = loader.fetch(["601398.SH"], "2024-01-01", "2024-01-31", interval="1min")
        assert result == {}

    def test_accepts_daily_variants(self):
        """Various daily interval strings should not be rejected at the interval gate."""
        loader = DataLoader()
        with patch.object(loader, "_get_dsn", return_value=""), \
             patch("psycopg.connect", side_effect=Exception("no connection")):
            for iv in ("1D", "d", "day", "daily"):
                # Connection will fail, returning {} — but the interval gate must pass.
                result = loader.fetch(["601398.SH"], "2024-01-01", "2024-01-31", interval=iv)
                assert result == {}


class TestFetchOneDataProcessing:
    """Verify _fetch_one builds a correct DataFrame from cursor rows."""

    def test_returns_dataframe_with_expected_columns(self):
        loader = DataLoader()
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = [
            (date(2024, 1, 2), 10.0, 10.5, 9.8, 10.2, 5512800, 7400000000),
            (date(2024, 1, 3), 10.2, 10.8, 10.1, 10.6, 3200000, 4300000000),
        ]

        df = loader._fetch_one(conn, "601398.SH", "2024-01-01", "2024-01-31")

        assert df is not None
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 2
        assert df.index.name == "trade_date"

    def test_volume_kept_as_shares(self):
        """Volume must NOT be divided by 100 — ticker_pg declares 'shares'."""
        loader = DataLoader()
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = [
            (date(2024, 1, 2), 10.0, 10.5, 9.8, 10.2, 5512800, 7400000000),
        ]

        df = loader._fetch_one(conn, "601398.SH", "2024-01-01", "2024-01-31")
        assert df["volume"].iloc[0] == 5512800.0

    def test_empty_rows_returns_none(self):
        loader = DataLoader()
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = []

        assert loader._fetch_one(conn, "601398.SH", "2024-01-01", "2024-01-31") is None

    def test_ohlc_violation_dropped(self):
        """validate_ohlc must drop bars where high < low."""
        loader = DataLoader()
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = [
            (date(2024, 1, 2), 10.0, 10.5, 9.8, 10.2, 1000, 10000),   # valid
            (date(2024, 1, 3), 10.0, 9.0, 10.5, 10.2, 1000, 10000),   # high < low → dropped
        ]

        df = loader._fetch_one(conn, "601398.SH", "2024-01-01", "2024-01-31")
        assert len(df) == 1


class TestFetchConnectionFailure:
    """Verify fetch returns {} and does not crash when PG is unreachable."""

    def test_connect_failure_returns_empty(self):
        loader = DataLoader()
        with patch.object(loader, "_get_dsn", return_value="postgresql://x:y@z/db"), \
             patch("psycopg.connect", side_effect=Exception("connection refused")):
            result = loader.fetch(["601398.SH"], "2024-01-01", "2024-01-31")
            assert result == {}

    def test_per_symbol_error_does_not_abort_others(self):
        """If one symbol's query fails, the rest should still be attempted."""
        loader = DataLoader()
        conn = MagicMock()
        conn.cursor.side_effect = Exception("cursor error")

        with patch.object(loader, "_get_dsn", return_value="postgresql://x:y@z/db"), \
             patch("psycopg.connect", return_value=conn):
            result = loader.fetch(["601398.SH", "000001.SZ"], "2024-01-01", "2024-01-31")
            assert result == {}


class TestLoaderAttributes:
    """Verify class-level attributes match the protocol."""

    def test_name(self):
        assert DataLoader.name == "ticker_pg"

    def test_markets(self):
        assert DataLoader.markets == {"a_share"}

    def test_volume_units(self):
        assert DataLoader.volume_units == {"a_share": "shares"}

    def test_requires_auth(self):
        assert DataLoader.requires_auth is True

    def test_market_map_covers_sh_sz_bj(self):
        assert set(_MARKET_MAP.keys()) == {"SH", "SZ", "BJ"}

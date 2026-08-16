"""Ticker PG loader: reads A-share daily bars from the local ticker PostgreSQL.

Reads pre-adjusted (qfq) daily OHLCV from the shared ``ticker`` database's
``v_daily_bar_qfq`` view.  Falls back through the external-source chain when
the PG DSN is unset or the database is unreachable.

Covers: A-shares (SH/SZ/BJ), daily bars only.  Volume is in single shares.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from backtest.loaders.base import (
    cached_loader_fetch,
    validate_date_range,
    validate_ohlc,
)
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

_DSN_ENV = "TICKER_PG_READONLY_DSN"

_MARKET_MAP = {"SH": "SH", "SZ": "SZ", "BJ": "BJ"}


@register
class DataLoader:
    """Ticker PG A-share OHLCV loader (read-only, daily only)."""

    name = "ticker_pg"
    markets = {"a_share"}
    volume_units = {"a_share": "shares"}
    requires_auth = True

    def __init__(self) -> None:
        self._dsn = __import__("os").environ.get(_DSN_ENV, "")

    def is_available(self) -> bool:
        if not self._dsn:
            return False
        try:
            import psycopg  # noqa: F401
            return True
        except ImportError:
            return False

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch A-share daily OHLCV from ticker PG ``v_daily_bar_qfq``.

        Args:
            codes: Symbol list (e.g. ["601398.SH", "000001.SZ"]).
            start_date: YYYY-MM-DD.
            end_date: YYYY-MM-DD.
            interval: Bar size (only 1D supported).
            fields: Ignored.

        Returns:
            Mapping symbol -> OHLCV DataFrame.
        """
        validate_date_range(start_date, end_date)

        if str(interval).strip().lower() not in {"1d", "d", "day", "daily"}:
            logger.warning(
                "ticker_pg supports daily bars only; rejecting interval=%r",
                interval,
            )
            return {}

        import psycopg

        try:
            conn = psycopg.connect(self._dsn, connect_timeout=6)
        except Exception as exc:
            logger.warning("ticker_pg connect failed: %s", exc)
            return {}

        result: Dict[str, pd.DataFrame] = {}
        try:
            for code in codes:
                try:
                    df = cached_loader_fetch(
                        source=self.name,
                        symbol=code,
                        timeframe=interval,
                        start_date=start_date,
                        end_date=end_date,
                        fields=None,
                        fetch=lambda code=code: self._fetch_one(
                            conn, code, start_date, end_date
                        ),
                    )
                    if df is not None and not df.empty:
                        result[code] = df
                except Exception as exc:
                    logger.warning("ticker_pg failed for %s: %s", code, exc)
        finally:
            conn.close()

        return result

    def _fetch_one(
        self, conn, code: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch a single A-share symbol from v_daily_bar_qfq."""
        parts = code.upper().split(".")
        if len(parts) != 2:
            return None
        symbol, market = parts[0], parts[1]
        if market not in _MARKET_MAP:
            return None

        query = """
            SELECT trade_date, open, high, low, close, volume, amount
            FROM v_daily_bar_qfq
            WHERE market = %s AND code = %s
              AND trade_date BETWEEN %s AND %s
            ORDER BY trade_date
        """
        df = pd.read_sql(query, conn, params=(market, symbol, start_date, end_date))

        if df.empty:
            return None

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.set_index("trade_date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].dropna(
            subset=["open", "high", "low", "close"]
        )
        df = validate_ohlc(df)
        return df

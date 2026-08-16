# vibe-Trading 数据源接入方案：添加 `ticker_pg` Loader

> **状态：已完成** — 2026-08-16 实施并验证通过，代码已推送至 `origin/main`（commit `2a173a3`、`c17a68e`）。

---

## 一、现状

### 1.1 数据源架构

vibe-Trading 使用 **Loader 注册 + 回退链** 架构：

- 每个数据源是一个独立的 `.py` 文件，通过 `@register` 装饰器注册到全局 `LOADER_REGISTRY`
- 每个市场（`a_share` / `us_equity` / `hk_equity` 等）有独立的回退链
- 回退链按优先级排列，`resolve_loader()` 返回第一个 `is_available()` 为 True 的 loader

### 1.2 A 股回退链现状

```
a_share: tencent → mootdx → eastmoney → baostock → akshare → tushare → local
```

**全部是外部免费源**，无局域网 PG 数据层。

### 1.3 关键文件

| 文件 | 用途 |
|------|------|
| `agent/backtest/loaders/base.py` | `DataLoaderProtocol` 接口定义 |
| `agent/backtest/loaders/registry.py` | `LOADER_REGISTRY`, `VALID_SOURCES`, `FALLBACK_CHAINS`, `_loader_modules` |
| `agent/backtest/loaders/baostock_loader.py` | 参考实现（A 股日线，no auth，简单） |
| `agent/src/config/env_schema.py` | `DataConfig` 环境变量 schema |
| `agent/.env.example` | 环境变量模板 |

---

## 二、方案：新增 `ticker_pg` Loader

### 2.1 设计决策

| 维度 | 决策 |
|------|------|
| 数据范围 | **A 股日线**（`v_daily_bar_qfq` 前复权） |
| 市场声明 | `markets = {"a_share"}` |
| 回退链位置 | **链首**（`ticker_pg` → tencent → mootdx → ...），PG 不可用时自动回退 |
| 复权处理 | 直接读 `v_daily_bar_qfq`（前复权），不自算复权因子 |
| 实时行情 | 不支持（`v_quote_rt` 是按需缓存，不符合 vibe-Trading 的实时需求） |
| 分钟线 | 不支持（共享库无分钟线） |
| 凭证 | 环境变量 `TICKER_PG_READONLY_DSN`（可选，不设则 `is_available()` 返回 False） |

### 2.2 改动文件清单

### 文件 1: `agent/backtest/loaders/ticker_pg_loader.py`（新建）

**Loader 实现**，参考 `baostock_loader.py` 模式：

```python
"""
Ticker PG loader: reads A-share daily bars from the local ticker PostgreSQL
database (v_daily_bar_qfq view). Falls back through the external-source chain
when the PG DSN is unset or the database is unreachable.

Covers: A-shares (SH/SZ/BJ), daily bars only.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import pandas as pd

from backtest.loaders.base import cached_loader_fetch, validate_date_range
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

_DSN_ENV = "TICKER_PG_READONLY_DSN"


@register
class DataLoader:
    """Ticker PG A-share OHLCV loader (read-only, daily only)."""

    name = "ticker_pg"
    markets = {"a_share"}
    volume_units = {"a_share": "shares"}  # v_daily_bar.volume = 股（单股）
    requires_auth = True

    def __init__(self) -> None:
        self._dsn = os.environ.get(_DSN_ENV, "")

    def is_available(self) -> bool:
        """Available if TICKER_PG_READONLY_DSN is set and psycopg is installed."""
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
        """Fetch A-share daily OHLCV from ticker PG v_daily_bar_qfq.

        Args:
            codes: Symbol list (e.g. ["601595.SH", "000001.SZ"]).
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
        # Parse market + code from tushare-style suffix (601398.SH / 000001.SZ)
        parts = code.upper().split(".")
        if len(parts) != 2:
            return None
        symbol, market = parts[0], parts[1]
        market_map = {"SH": "SH", "SZ": "SZ", "BJ": "BJ"}
        if market not in market_map:
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
        return df
```

### 文件 2: `agent/backtest/loaders/registry.py`（修改）

三处改动：

**① `VALID_SOURCES` 添加 `"ticker_pg"`（第 33-59 行）**

```python
VALID_SOURCES: set[str] = {
    "tushare", "okx", "binance", "yfinance", "akshare", "baostock",
    "tencent", "mootdx", "ccxt", "futu", "eastmoney", "sina", "stooq",
    "yahoo", "finnhub", "alphavantage", "tiingo", "fmp", "qveris",
    "india_broker", "pykrx", "longbridge", "mt5", "local", "auto",
    "ticker_pg",  # ← 新增
}
```

**② `_loader_modules` 添加模块路径（第 83-108 行）**

```python
_loader_modules = [
    "backtest.loaders.tushare",
    ...
    "backtest.loaders.local_loader",
    "backtest.loaders.ticker_pg_loader",  # ← 新增
]
```

**③ `FALLBACK_CHAINS` 的 `a_share` 链首插入 `"ticker_pg"`（第 137 行）**

```python
"a_share":   ["ticker_pg", "tencent", "mootdx", "eastmoney", "baostock", "akshare", "tushare", "local"],
```

### 文件 3: `agent/src/config/env_schema.py`（修改）

在 `DataConfig` 类中新增字段（第 153-197 行，现有字段后）：

```python
ticker_pg_readonly_dsn: str = Field(alias="TICKER_PG_READONLY_DSN", default="")
```

### 文件 4: `agent/.env.example`（修改）

在 `## Data Sources` 段落（第 152-180 行）添加：

```env
# Ticker PG (local read-only PostgreSQL data layer, A-share daily bars only)
# DSN format: postgresql://<app>_reader:<password>@192.168.199.181:5432/ticker
# TICKER_PG_READONLY_DSN=postgresql://vibe_trading_reader:<password>@192.168.199.181:5432/ticker
```

### 文件 5: `agent/.env`（修改，实际环境配置）

项目实际 `.env` 文件中添加：

```env
TICKER_PG_READONLY_DSN=postgresql://vibe_trading_reader:<password>@192.168.199.181:5432/ticker
```

---

## 三、接入补偿

### 3.1 限制说明

| 功能 | ticker_pg 支持 | 替代方案 |
|------|---------------|----------|
| A 股日线 OHLCV | ✅ `v_daily_bar_qfq` | 首选 |
| A 股前复权日线 | ✅ `v_daily_bar_qfq` | 首选 |
| A 股后复权日线 | ✅ `v_daily_bar_hfq` | 可扩展 |
| A 股未复权日线 | ✅ `v_daily_bar` | 可扩展 |
| A 股分钟线 | ❌ 无 | mootdx/eastmoney 回退 |
| A 股实时行情 | ❌ 无 | tencent 回退 |
| A 股财务数据 | ✅ `v_financial_fundamental` | 可扩展（二期） |
| A 股交易日历 | ✅ `v_trade_calendar` | 可扩展（二期） |
| 港股/美股/加密 | ❌ 无 | 现有源不变 |
| 复权因子 | ✅ `v_adjust_factor` | 可扩展 |

### 3.2 凭证申请

需要向运维申请 `vibe_trading_reader` 只读角色：

```sql
-- 运维侧执行
CREATE ROLE vibe_trading_reader WITH LOGIN PASSWORD '<generated_password>';
GRANT USAGE ON SCHEMA public TO vibe_trading_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO vibe_trading_reader;
```

### 3.3 依赖

在 `pyproject.toml` 或 `requirements-lock.txt` 中检查 `psycopg[binary]` 是否已安装。可在 loader 的 `is_available()` 中惰性检测（代码已实现）。

---

## 四、回退行为

PG 不可用时，loader 自动降级：

```
ticker_pg (DSN 未设置/连接失败/查询异常)
  → 返回空 dict {}
  → resolve_loader() 继续走 chain 下一个源
  → tencent → mootdx → eastmoney → baostock → ...
```

**零影响**：现有回退链完全保留，PG 只是增加一个首选源。

---

## 五、验证清单

1. ✅ 新增 `ticker_pg_loader.py`，实现 `DataLoaderProtocol`
2. ✅ `registry.py` 更新 `VALID_SOURCES`、`_loader_modules`、`FALLBACK_CHAINS`
3. ✅ `env_schema.py` 添加 `TICKER_PG_READONLY_DSN`
4. ✅ `.env.example` 添加说明
5. ✅ 运行测试：`python -m pytest tests/ -k "test_valid_sources"` 验证 source 覆盖
6. ✅ 手动测试：`resolve_loader("a_share")` 返回 ticker_pg 实例
7. ✅ 手动测试：PG 不可用时自动回退到 tencent

---

## 六、工作量估算

| 步骤 | 文件 | 预估时间 |
|------|------|---------|
| 新建 loader | `ticker_pg_loader.py` | ~30 分钟 |
| 注册 loader | `registry.py`（3 处改动） | ~5 分钟 |
| 添加 env 变量 | `env_schema.py` + `.env.example` + `.env` | ~5 分钟 |
| 测试 | 运行验证 | ~10 分钟 |
| **合计** | **5 个文件改动** | **~50 分钟** |
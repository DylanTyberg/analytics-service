"""
Thin Polygon.io client for daily aggregate bars.

Isolated from the management command so it can be mocked in tests without
touching the network, and so the HTTP/parsing concerns live in one place.
Returns plain dataclasses -- the command decides how to persist them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal

import requests

BASE_URL = "https://api.polygon.io"


class PolygonError(Exception):
    pass


@dataclass(frozen=True)
class Bar:
    bar_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    adj_close: Decimal
    volume: int


def _to_decimal(value) -> Decimal:
    # Round to the 4dp the schema stores; Polygon can return long floats.
    return Decimal(str(value)).quantize(Decimal("0.0001"))


class PolygonClient:
    def __init__(self, api_key: str, *, session: requests.Session | None = None,
                 timeout: int = 30):
        if not api_key:
            raise PolygonError("POLYGON_API_KEY is not set")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.timeout = timeout

    def daily_bars(self, symbol: str, start: date, end: date) -> list[Bar]:
        """
        Fetch the full [start, end] range in one adjusted request.

        Fetching the whole window at once is deliberate: Polygon adjusts
        prices as of request time, so pulling everything in a single call
        guarantees one consistent split-adjustment basis across the series.
        Incremental appends would mix bases and manufacture phantom returns.

        The endpoint returns unadjusted OHLC by default; adjusted=true gives
        split/dividend-adjusted values. We store the adjusted close as
        adj_close and keep the (also adjusted) OHLC for display.
        """
        url = (
            f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{start.isoformat()}/{end.isoformat()}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,   # ~137 years of daily bars; one page is plenty
            "apiKey": self.api_key,
        }

        resp = self.session.get(url, params=params, timeout=self.timeout)

        if resp.status_code == 429:
            raise PolygonError(f"rate limited on {symbol}")
        if resp.status_code != 200:
            raise PolygonError(f"{symbol}: HTTP {resp.status_code} {resp.text[:200]}")

        payload = resp.json()
        status = payload.get("status")
        if status not in ("OK", "DELAYED"):
            raise PolygonError(f"{symbol}: status {status}")

        results = payload.get("results") or []
        bars = []
        for r in results:
            # 't' is epoch milliseconds at market open, UTC.
            bar_dt = datetime.fromtimestamp(r["t"] / 1000, tz=timezone.utc).date()
            bars.append(Bar(
                bar_date=bar_dt,
                open=_to_decimal(r["o"]),
                high=_to_decimal(r["h"]),
                low=_to_decimal(r["l"]),
                close=_to_decimal(r["c"]),
                adj_close=_to_decimal(r["c"]),  # adjusted=true -> c is adjusted
                volume=int(r["v"]),
            ))
        return bars
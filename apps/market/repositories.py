"""
Data access for price history.

This is the seam between SQL and pandas. Every function here takes query
parameters and returns a DataFrame or plain rows -- the analytics engine
never touches the ORM or a cursor, and the ORM models never grow analytics
methods. One module owns "SQL in, DataFrame out".

Why raw SQL here rather than the ORM: these are the analytical queries --
pivots, window functions, multi-symbol range scans over the largest table.
The ORM can express some of them, but the SQL is clearer, tunable with
EXPLAIN ANALYZE, and honest about what actually runs. The application-layer
queries (a user's holdings, a single security) stay in the ORM where its
safety and convenience are worth more than control.

All queries are parameterised. Never interpolate a value into the SQL
string -- psycopg's %(name)s placeholders are the injection boundary.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from django.db import connection


def price_frame(
    symbols: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Adjusted closes for several symbols, pivoted into the wide frame the
    engine expects: a DateTimeIndex with one column per symbol.

    The pivot happens in Postgres-adjacent pandas rather than via SQL
    crosstab because the symbol set is dynamic -- a fixed-column crosstab
    would need the columns known at query time. Pulling long and pivoting
    in pandas keeps the query simple and the column set data-driven.

    adj_close, never close: returns computed from unadjusted prices treat
    a split as a real move. This is the single most important line in the
    module.
    """
    if not symbols:
        return pd.DataFrame()

    sql = """
        SELECT b.bar_date, s.symbol, b.adj_close
        FROM daily_bars b
        JOIN securities s ON s.id = b.security_id
        WHERE s.symbol = ANY(%(symbols)s)
          AND b.bar_date BETWEEN %(start)s AND %(end)s
        ORDER BY b.bar_date
    """

    with connection.cursor() as cur:
        cur.execute(sql, {"symbols": symbols, "start": start, "end": end})
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    long_df = pd.DataFrame(rows, columns=["bar_date", "symbol", "adj_close"])
    # adj_close arrives as Decimal; the engine wants float. Convert once,
    # here, at the boundary -- not scattered through the math.
    long_df["adj_close"] = long_df["adj_close"].astype(float)

    wide = long_df.pivot(index="bar_date", columns="symbol", values="adj_close")
    wide.index = pd.to_datetime(wide.index)
    wide.columns.name = None
    return wide.sort_index()


def latest_prices(symbols: list[str]) -> dict[str, float]:
    """
    Most recent adjusted close per symbol.

    DISTINCT ON is the Postgres-idiomatic "latest row per group": order by
    symbol then date descending, and take the first row of each symbol
    group. Cleaner and faster than a correlated subquery or a window-plus-
    filter, and it uses the (security, bar_date DESC) index directly.
    """
    if not symbols:
        return {}

    sql = """
        SELECT DISTINCT ON (s.symbol) s.symbol, b.adj_close
        FROM daily_bars b
        JOIN securities s ON s.id = b.security_id
        WHERE s.symbol = ANY(%(symbols)s)
        ORDER BY s.symbol, b.bar_date DESC
    """

    with connection.cursor() as cur:
        cur.execute(sql, {"symbols": symbols})
        rows = cur.fetchall()

    return {symbol: float(price) for symbol, price in rows}


def coverage(symbols: list[str]) -> dict[str, dict]:
    """
    First date, last date and bar count per symbol.

    Used before an analysis to decide which positions have enough history
    to include, and by the ingest command to know where to resume. Returns
    a plain dict so callers don't need pandas for a cheap metadata check.
    """
    if not symbols:
        return {}

    sql = """
        SELECT s.symbol,
               MIN(b.bar_date) AS first_date,
               MAX(b.bar_date) AS last_date,
               COUNT(*)        AS bar_count
        FROM daily_bars b
        JOIN securities s ON s.id = b.security_id
        WHERE s.symbol = ANY(%(symbols)s)
        GROUP BY s.symbol
    """

    with connection.cursor() as cur:
        cur.execute(sql, {"symbols": symbols})
        rows = cur.fetchall()

    return {
        symbol: {"first_date": first, "last_date": last, "bar_count": count}
        for symbol, first, last, count in rows
    }
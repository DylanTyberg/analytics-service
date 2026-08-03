"""
Data access for trade lots.

Same contract as market/repositories.py: SQL in, plain rows or engine
dataclasses out. Nothing here imports pandas -- lots are objects, not a
matrix.

This module is the clearest justification in the project for a relational
store. The queries below -- matching, hold-period aggregation partitioned
by outcome, running P&L -- are exactly what would be painful over a
key-value log. In Postgres they are a GROUP BY and a window function.
"""

from __future__ import annotations

from decimal import Decimal

from django.db import connection

from apps.analytics.engine.behavior import ClosedLot, OpenLot


def closed_lots(user_id: int) -> list[ClosedLot]:
    """
    All closed lots for a user, as engine dataclasses ready for the
    behavioural functions.

    Returns the pure ClosedLot type, not the Django model -- the engine
    stays ignorant of the ORM, and this function is the one place that
    knows both sides.
    """
    sql = """
        SELECT s.symbol, l.original_qty, l.open_price, l.close_price,
               l.opened_at, l.closed_at, l.realized_pnl, l.hold_days
        FROM position_lots l
        JOIN securities s ON s.id = l.security_id
        WHERE l.user_id = %(user_id)s
          AND l.closed_at IS NOT NULL
        ORDER BY l.closed_at
    """

    with connection.cursor() as cur:
        cur.execute(sql, {"user_id": user_id})
        rows = cur.fetchall()

    return [
        ClosedLot(
            symbol=symbol,
            quantity=qty,
            open_price=open_price,
            close_price=close_price,
            opened_at=opened_at,
            closed_at=closed_at,
            realized_pnl=pnl,
            hold_days=hold_days,
        )
        for (symbol, qty, open_price, close_price,
             opened_at, closed_at, pnl, hold_days) in rows
    ]


def open_lots(user_id: int) -> list[OpenLot]:
    """Open lots for the activity and turnover calculations."""
    sql = """
        SELECT s.symbol, l.remaining_qty, l.open_price, l.opened_at
        FROM position_lots l
        JOIN securities s ON s.id = l.security_id
        WHERE l.user_id = %(user_id)s
          AND l.closed_at IS NULL
        ORDER BY l.opened_at
    """

    with connection.cursor() as cur:
        cur.execute(sql, {"user_id": user_id})
        rows = cur.fetchall()

    return [
        OpenLot(symbol=symbol, quantity=qty, open_price=price, opened_at=opened_at)
        for symbol, qty, price, opened_at in rows
    ]


def current_holdings(user_id: int) -> dict[str, Decimal]:
    """
    Net open quantity per symbol, summed across lots.

    These are the weights the risk metrics run on. Aggregating open lots
    rather than reading a separate holdings table keeps Postgres consistent
    with the lots that actually exist here -- the DynamoDB holdings item is
    the app's source of truth, but this side derives weights from its own
    data so the two can be reconciled.
    """
    sql = """
        SELECT s.symbol, SUM(l.remaining_qty) AS qty
        FROM position_lots l
        JOIN securities s ON s.id = l.security_id
        WHERE l.user_id = %(user_id)s
          AND l.closed_at IS NULL
        GROUP BY s.symbol
        HAVING SUM(l.remaining_qty) > 0
    """

    with connection.cursor() as cur:
        cur.execute(sql, {"user_id": user_id})
        rows = cur.fetchall()

    return {symbol: qty for symbol, qty in rows}


def hold_period_by_outcome(user_id: int) -> dict:
    """
    The disposition effect computed IN SQL, as a cross-check on the engine.

    The engine computes this from ClosedLot objects; computing it again in
    the database over the same data is a cheap correctness check, and it is
    also the query worth showing when the conversation turns to SQL. A
    single GROUP BY over a boolean (win vs loss) yields both averages that
    the ratio is built from.

    FILTER is the clean Postgres way to aggregate conditionally without
    CASE-inside-AVG contortions.
    """
    sql = """
        SELECT
            AVG(hold_days) FILTER (WHERE realized_pnl > 0)  AS avg_winner_hold,
            AVG(hold_days) FILTER (WHERE realized_pnl <= 0) AS avg_loser_hold,
            COUNT(*)       FILTER (WHERE realized_pnl > 0)  AS n_winners,
            COUNT(*)       FILTER (WHERE realized_pnl <= 0) AS n_losers
        FROM position_lots
        WHERE user_id = %(user_id)s
          AND closed_at IS NOT NULL
    """

    with connection.cursor() as cur:
        cur.execute(sql, {"user_id": user_id})
        row = cur.fetchone()

    avg_win, avg_loss, n_win, n_loss = row
    ratio = None
    if avg_win and avg_win > 0 and avg_loss is not None:
        ratio = float(avg_loss) / float(avg_win)

    return {
        "avg_winner_hold": float(avg_win) if avg_win is not None else None,
        "avg_loser_hold": float(avg_loss) if avg_loss is not None else None,
        "n_winners": n_win,
        "n_losers": n_loss,
        "ratio": ratio,
    }


def realized_pnl_timeline(user_id: int) -> list[dict]:
    """
    Cumulative realised P&L over time, one row per closed lot.

    The window function -- SUM over an ordered frame -- is the running
    total. This is the equity-curve data for the UI, and the canonical
    "why relational" example: a running aggregate that would need
    application-side accumulation over a key-value store is one line here.
    """
    sql = """
        SELECT
            l.closed_at::date AS on_date,
            l.realized_pnl,
            SUM(l.realized_pnl) OVER (
                ORDER BY l.closed_at
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_pnl
        FROM position_lots l
        WHERE l.user_id = %(user_id)s
          AND l.closed_at IS NOT NULL
        ORDER BY l.closed_at
    """

    with connection.cursor() as cur:
        cur.execute(sql, {"user_id": user_id})
        rows = cur.fetchall()

    return [
        {
            "date": on_date,
            "realized_pnl": float(pnl),
            "cumulative_pnl": float(cumulative),
        }
        for on_date, pnl, cumulative in rows
    ]
"""
Behavioural analytics.

Computed from closed position lots -- the output of FIFO matching. This is
the half of the feature that no other portfolio project has, because it
needs trade history that only this platform generates.

Same purity contract as the rest of engine/: plain dataclasses in, plain
dataclasses out, no Django and no database. The Django wrapper that loads
lots and persists results lives in services.py.

Everything here is descriptive and backward-looking. It measures what the
user actually did -- how long they held, how often they won, whether they
beat simply holding. It predicts nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal


@dataclass(frozen=True)
class ClosedLot:
    """
    Minimal view of a closed lot needed for the metrics. Decoupled from
    the Django model so this module stays pure and testable.
    """

    symbol: str
    quantity: Decimal
    open_price: Decimal
    close_price: Decimal
    opened_at: datetime
    closed_at: datetime
    realized_pnl: Decimal
    hold_days: int

    @property
    def is_winner(self) -> bool:
        return self.realized_pnl > 0

    @property
    def cost_basis(self) -> Decimal:
        return self.open_price * self.quantity

    @property
    def return_pct(self) -> Decimal:
        if self.open_price == 0:
            return Decimal("0")
        return (self.close_price - self.open_price) / self.open_price


@dataclass(frozen=True)
class OpenLot:
    """Open lots contribute to turnover and buy-hold, not to closed-lot stats."""

    symbol: str
    quantity: Decimal
    open_price: Decimal
    opened_at: datetime


@dataclass
class DispositionResult:
    """
    The disposition effect: the tendency to sell winners early and hold
    losers too long. Ratio > 1 means losers were held longer.
    """

    avg_hold_winners: float | None
    avg_hold_losers: float | None
    ratio: float | None
    n_winners: int
    n_losers: int


@dataclass
class OutcomeResult:
    total_closed: int
    winners: int
    win_rate: float | None
    avg_win_pct: float | None
    avg_loss_pct: float | None
    payoff_ratio: float | None       # avg win / |avg loss|
    # Combines both into the number that actually predicts profitability:
    # win_rate * avg_win must exceed loss_rate * avg_loss to make money.
    expectancy: float | None


@dataclass
class ActivityResult:
    total_closed_lots: int
    span_days: int
    trades_per_month: float | None
    turnover_ratio: float | None


@dataclass
class PerformanceResult:
    """Did the user's timing beat simply having held the same positions?"""

    realized_return: float | None
    buy_hold_return: float | None
    vs_buy_hold: float | None        # realized - buy_hold; negative is common


# ---------------------------------------------------------------------
# Disposition effect
# ---------------------------------------------------------------------


def disposition_effect(lots: list[ClosedLot]) -> DispositionResult:
    """
    Compare average hold time of winners against losers.

    The classic behavioural-finance finding is that retail traders hold
    losers longer than winners (ratio > 1) -- realising gains quickly for
    the satisfaction while avoiding the regret of locking in a loss. A
    disciplined trader shows a ratio near or below 1.
    """
    winners = [l for l in lots if l.is_winner]
    losers = [l for l in lots if not l.is_winner]

    avg_win = _mean(l.hold_days for l in winners) if winners else None
    avg_loss = _mean(l.hold_days for l in losers) if losers else None

    ratio = None
    if avg_win and avg_win > 0 and avg_loss is not None:
        ratio = avg_loss / avg_win

    return DispositionResult(
        avg_hold_winners=avg_win,
        avg_hold_losers=avg_loss,
        ratio=ratio,
        n_winners=len(winners),
        n_losers=len(losers),
    )


# ---------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------


def outcomes(lots: list[ClosedLot]) -> OutcomeResult:
    """
    Win rate, average win/loss size, and the ratios that combine them.

    Win rate alone is a trap: a 30% win rate is highly profitable if wins
    are far larger than losses. Payoff ratio and expectancy are what
    actually matter, which is why both are computed here.
    """
    if not lots:
        return OutcomeResult(0, 0, None, None, None, None, None)

    winners = [l for l in lots if l.is_winner]
    losers = [l for l in lots if not l.is_winner]
    n = len(lots)

    win_rate = len(winners) / n
    avg_win = _mean(float(l.return_pct) for l in winners) if winners else None
    avg_loss = _mean(float(l.return_pct) for l in losers) if losers else None

    payoff = None
    if avg_win is not None and avg_loss not in (None, 0):
        payoff = avg_win / abs(avg_loss)

    # Expectancy per trade in return terms. Positive means the strategy
    # makes money on average even if the win rate is low.
    expectancy = None
    if avg_win is not None and avg_loss is not None:
        loss_rate = 1.0 - win_rate
        expectancy = win_rate * avg_win + loss_rate * avg_loss

    return OutcomeResult(
        total_closed=n,
        winners=len(winners),
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        payoff_ratio=payoff,
        expectancy=expectancy,
    )


# ---------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------


def activity(
    lots: list[ClosedLot],
    open_lots: list[OpenLot],
    portfolio_value: float | None = None,
) -> ActivityResult:
    """
    Trading frequency and turnover.

    Turnover is total traded value relative to portfolio size -- a proxy
    for how much churn the account sees. High turnover is associated with
    worse net returns after costs, one of the most robust findings in the
    retail-trading literature.
    """
    if not lots:
        return ActivityResult(0, 0, None, None)

    opens = [l.opened_at for l in lots] + [l.opened_at for l in open_lots]
    closes = [l.closed_at for l in lots]
    first = min(opens)
    last = max(closes)
    span_days = max((last - first).days, 1)

    months = span_days / 30.44
    trades_per_month = len(lots) / months if months > 0 else None

    turnover = None
    if portfolio_value and portfolio_value > 0:
        # Round-trip notional: buy side + sell side of each closed lot.
        traded = sum(
            float(l.cost_basis) + float(l.close_price * l.quantity) for l in lots
        )
        turnover = traded / portfolio_value

    return ActivityResult(
        total_closed_lots=len(lots),
        span_days=span_days,
        trades_per_month=trades_per_month,
        turnover_ratio=turnover,
    )


# ---------------------------------------------------------------------
# Performance vs buy-and-hold
# ---------------------------------------------------------------------


def performance_vs_buy_hold(
    lots: list[ClosedLot],
    current_prices: dict[str, Decimal],
) -> PerformanceResult:
    """
    The honest scorecard: did active trading beat holding?

    Realised return is what the user actually made on closed lots. Buy-hold
    is what those same lots would have returned if never sold -- valued at
    the current price instead of the sale price. When vs_buy_hold is
    negative, the user's timing cost them money, which is the common and
    useful finding.

    Both are dollar-weighted (aggregate P&L over aggregate cost basis), so
    larger positions count more -- the same way the account actually felt
    the outcomes.
    """
    if not lots:
        return PerformanceResult(None, None, None)

    total_cost = sum(float(l.cost_basis) for l in lots)
    if total_cost == 0:
        return PerformanceResult(None, None, None)

    realized_pnl = sum(float(l.realized_pnl) for l in lots)
    realized_return = realized_pnl / total_cost

    # Buy-hold: value each lot at its symbol's current price. Lots whose
    # symbol has no current price are skipped on both sides so the two
    # numbers stay comparable.
    bh_cost = 0.0
    bh_pnl = 0.0
    for l in lots:
        price = current_prices.get(l.symbol)
        if price is None:
            continue
        cost = float(l.cost_basis)
        value = float(price) * float(l.quantity)
        bh_cost += cost
        bh_pnl += value - cost

    buy_hold_return = bh_pnl / bh_cost if bh_cost > 0 else None

    vs = (
        realized_return - buy_hold_return
        if buy_hold_return is not None
        else None
    )

    return PerformanceResult(
        realized_return=realized_return,
        buy_hold_return=buy_hold_return,
        vs_buy_hold=vs,
    )


# ---------------------------------------------------------------------
# Distributions (for charting)
# ---------------------------------------------------------------------


def hold_period_buckets(lots: list[ClosedLot]) -> dict[str, int]:
    """Count closed lots by holding-period band. Presentation data."""
    buckets = {
        "0-1d": 0, "2-7d": 0, "8-30d": 0, "31-90d": 0, "91-365d": 0, "365d+": 0,
    }
    for l in lots:
        d = l.hold_days
        if d <= 1:
            buckets["0-1d"] += 1
        elif d <= 7:
            buckets["2-7d"] += 1
        elif d <= 30:
            buckets["8-30d"] += 1
        elif d <= 90:
            buckets["31-90d"] += 1
        elif d <= 365:
            buckets["91-365d"] += 1
        else:
            buckets["365d+"] += 1
    return buckets


def monthly_trade_counts(lots: list[ClosedLot]) -> dict[str, int]:
    """Closed lots per calendar month (keyed 'YYYY-MM'). Presentation data."""
    counts: dict[str, int] = {}
    for l in lots:
        key = l.closed_at.strftime("%Y-%m")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
"""
Tests for the behavioural engine.

Pure unit tests -- hand-built ClosedLot lists with worked-out expected
values. The important property throughout: a metric that ignored the
distinction it is supposed to measure would fail these, not merely differ
from a snapshot.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from apps.analytics.engine import behavior as bh
from apps.analytics.engine.behavior import ClosedLot, OpenLot

BASE = datetime(2026, 1, 1, 15, 0)


def closed(symbol, qty, open_p, close_p, hold_days, opened_offset=0):
    opened = BASE + timedelta(days=opened_offset)
    closed_at = opened + timedelta(days=hold_days)
    qty_d = Decimal(str(qty))
    op = Decimal(str(open_p))
    cp = Decimal(str(close_p))
    return ClosedLot(
        symbol=symbol,
        quantity=qty_d,
        open_price=op,
        close_price=cp,
        opened_at=opened,
        closed_at=closed_at,
        realized_pnl=(cp - op) * qty_d,
        hold_days=hold_days,
    )


# ---------------------------------------------------------------------
# Disposition effect -- the headline metric
# ---------------------------------------------------------------------


def test_disposition_detects_holding_losers_longer():
    """
    Winners held 10 days, losers held 20. Ratio must be 2.0. A function
    that ignored the winner/loser split would return ~1.0 and fail here --
    which is the whole point of this test.
    """
    lots = [
        closed("A", 10, 100, 110, hold_days=10),   # winner, 10d
        closed("B", 10, 100, 120, hold_days=10),   # winner, 10d
        closed("C", 10, 100, 90, hold_days=20),    # loser, 20d
        closed("D", 10, 100, 80, hold_days=20),    # loser, 20d
    ]
    result = bh.disposition_effect(lots)

    assert result.avg_hold_winners == pytest.approx(10.0)
    assert result.avg_hold_losers == pytest.approx(20.0)
    assert result.ratio == pytest.approx(2.0)
    assert result.n_winners == 2
    assert result.n_losers == 2


def test_disposition_disciplined_trader_below_one():
    """Cutting losers fast and letting winners run -> ratio < 1."""
    lots = [
        closed("A", 10, 100, 130, hold_days=40),   # winner held long
        closed("B", 10, 100, 90, hold_days=3),     # loser cut fast
    ]
    result = bh.disposition_effect(lots)

    assert result.ratio == pytest.approx(3 / 40)


def test_disposition_all_winners_gives_no_ratio():
    """No losers means the ratio is undefined, not zero or a crash."""
    lots = [closed("A", 10, 100, 110, hold_days=5)]
    result = bh.disposition_effect(lots)

    assert result.n_losers == 0
    assert result.ratio is None
    assert result.avg_hold_losers is None


# ---------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------


def test_win_rate_and_counts():
    lots = [
        closed("A", 10, 100, 110, 5),   # win
        closed("B", 10, 100, 105, 5),   # win
        closed("C", 10, 100, 95, 5),    # loss
    ]
    result = bh.outcomes(lots)

    assert result.total_closed == 3
    assert result.winners == 2
    assert result.win_rate == pytest.approx(2 / 3)


def test_payoff_ratio():
    """
    Avg win +20%, avg loss -10% -> payoff 2.0. This is what makes a low
    win rate survivable.
    """
    lots = [
        closed("A", 10, 100, 120, 5),   # +20%
        closed("B", 10, 100, 90, 5),    # -10%
    ]
    result = bh.outcomes(lots)

    assert result.avg_win_pct == pytest.approx(0.20)
    assert result.avg_loss_pct == pytest.approx(-0.10)
    assert result.payoff_ratio == pytest.approx(2.0)


def test_expectancy_positive_despite_low_win_rate():
    """
    One +50% win against two -10% losses: win rate is only 33% but
    expectancy is positive, because the win is large.
    (1/3)(0.50) + (2/3)(-0.10) = 0.1667 - 0.0667 = +0.10
    """
    lots = [
        closed("A", 10, 100, 150, 5),   # +50%
        closed("B", 10, 100, 90, 5),    # -10%
        closed("C", 10, 100, 90, 5),    # -10%
    ]
    result = bh.outcomes(lots)

    assert result.win_rate == pytest.approx(1 / 3)
    assert result.expectancy == pytest.approx(0.10)


def test_outcomes_empty():
    result = bh.outcomes([])
    assert result.total_closed == 0
    assert result.win_rate is None
    assert result.expectancy is None


# ---------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------


def test_trades_per_month():
    """10 closed lots spanning ~2 months -> ~5/month."""
    lots = [closed(f"S{i}", 10, 100, 105, 2, opened_offset=i * 6) for i in range(10)]
    result = bh.activity(lots, open_lots=[])

    assert result.total_closed_lots == 10
    assert result.trades_per_month == pytest.approx(5.0, rel=0.15)


def test_turnover_ratio():
    """
    One lot: buy 10 @ 100 (=1000), sell 10 @ 110 (=1100). Round-trip
    notional 2100 against a 10000 portfolio -> 0.21.
    """
    lots = [closed("A", 10, 100, 110, 5)]
    result = bh.activity(lots, open_lots=[], portfolio_value=10000.0)

    assert result.turnover_ratio == pytest.approx(0.21)


def test_turnover_none_without_portfolio_value():
    lots = [closed("A", 10, 100, 110, 5)]
    result = bh.activity(lots, open_lots=[])
    assert result.turnover_ratio is None


# ---------------------------------------------------------------------
# Performance vs buy-and-hold
# ---------------------------------------------------------------------


def test_vs_buy_hold_negative_when_sold_too_early():
    """
    Bought at 100, sold at 110 (+10% realised). Now trades at 150, so
    holding would have made +50%. The user's timing cost 40 points.
    """
    lots = [closed("A", 10, 100, 110, 5)]
    result = bh.performance_vs_buy_hold(lots, {"A": Decimal("150")})

    assert result.realized_return == pytest.approx(0.10)
    assert result.buy_hold_return == pytest.approx(0.50)
    assert result.vs_buy_hold == pytest.approx(-0.40)


def test_vs_buy_hold_positive_when_sold_before_drop():
    """Sold at a gain; the stock then fell below cost. Timing helped."""
    lots = [closed("A", 10, 100, 110, 5)]
    result = bh.performance_vs_buy_hold(lots, {"A": Decimal("80")})

    assert result.realized_return == pytest.approx(0.10)
    assert result.buy_hold_return == pytest.approx(-0.20)
    assert result.vs_buy_hold == pytest.approx(0.30)


def test_vs_buy_hold_dollar_weighted():
    """
    Larger positions count more. A big well-timed trade should dominate a
    small mistimed one in the aggregate.
    """
    lots = [
        closed("BIG", 100, 100, 110, 5),   # cost 10000, realised +10%
        closed("SML", 1, 100, 110, 5),     # cost 100, realised +10%
    ]
    # BIG now at 105 (holding would give +5%), SML at 200 (+100%).
    result = bh.performance_vs_buy_hold(lots, {"BIG": Decimal("105"), "SML": Decimal("200")})

    # Buy-hold dollar-weighted: (10000*0.05 + 100*1.00) / 10100
    expected_bh = (10000 * 0.05 + 100 * 1.00) / 10100
    assert result.buy_hold_return == pytest.approx(expected_bh)


def test_vs_buy_hold_skips_missing_prices():
    """A symbol with no current price drops out of both sides, not one."""
    lots = [
        closed("A", 10, 100, 110, 5),
        closed("B", 10, 100, 110, 5),
    ]
    result = bh.performance_vs_buy_hold(lots, {"A": Decimal("150")})  # B missing

    # Realised uses all lots; buy-hold uses only A. Both are defined.
    assert result.realized_return is not None
    assert result.buy_hold_return == pytest.approx(0.50)


# ---------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------


def test_hold_period_buckets():
    lots = [
        closed("A", 1, 100, 101, 0),
        closed("B", 1, 100, 101, 5),
        closed("C", 1, 100, 101, 15),
        closed("D", 1, 100, 101, 200),
    ]
    buckets = bh.hold_period_buckets(lots)

    assert buckets["0-1d"] == 1
    assert buckets["2-7d"] == 1
    assert buckets["8-30d"] == 1
    assert buckets["91-365d"] == 1
    assert buckets["31-90d"] == 0


# ---------------------------------------------------------------------
# Integration with seeded data
# ---------------------------------------------------------------------

from django.test import override_settings

@pytest.mark.django_db
@override_settings(DEBUG=True)
def test_seeded_population_shows_disposition_effect():
    """
    End-to-end against the seeder: generate a population with a known
    disposition ratio and confirm the engine recovers it. This is the
    payoff for building the seeder to have known properties.
    """
    from django.core.management import call_command
    from apps.trading.models import AppUser, PositionLot

    call_command(
        "seed_trades", user="pytest-demo", lots=300,
        disposition=1.6, seed=7, verbosity=0,
    )

    user = AppUser.objects.get(cognito_sub="pytest-demo")
    rows = PositionLot.objects.filter(user=user, closed_at__isnull=False)

    lots = [
        ClosedLot(
            symbol=r.security.symbol, quantity=r.original_qty,
            open_price=r.open_price, close_price=r.close_price,
            opened_at=r.opened_at, closed_at=r.closed_at,
            realized_pnl=r.realized_pnl, hold_days=r.hold_days,
        )
        for r in rows.select_related("security")
    ]

    result = bh.disposition_effect(lots)

    # The engine should recover a ratio in the neighbourhood of the 1.6
    # target -- loose bound because it is a sampled population.
    assert result.ratio == pytest.approx(1.6, abs=0.4)
    assert result.n_winners > 0 and result.n_losers > 0
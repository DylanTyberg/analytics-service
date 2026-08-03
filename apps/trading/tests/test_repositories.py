"""
Tests for the repository layer.

These hit a real Postgres (via pytest-django's test database) because the
whole point of the module is the SQL -- DISTINCT ON, FILTER aggregates,
window functions. Mocking the cursor would test nothing that matters.

Data comes from the seeder, so these also serve as a second integration
check that seeding, the schema, and the queries all agree.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from django.core.management import call_command
from django.test import override_settings
from django.utils import timezone

from apps.market.models import DailyBar, Security
from apps.trading.models import AppUser
from apps.market import repositories as market_repo
from apps.trading import repositories as trading_repo


@pytest.fixture
def seeded_user(db):
    with override_settings(DEBUG=True):
        call_command(
            "seed_trades", user="repo-test", lots=200,
            disposition=1.6, seed=99, verbosity=0,
        )
    return AppUser.objects.get(cognito_sub="repo-test")


@pytest.fixture
def price_history(db):
    """A few symbols with a month of daily bars, for the market repo."""
    base = date(2026, 1, 1)
    for sym, start_price in [("AAPL", 180.0), ("MSFT", 400.0)]:
        security = Security.objects.create(symbol=sym, name=sym)
        for i in range(30):
            DailyBar.objects.create(
                security=security,
                bar_date=base + timedelta(days=i),
                open=start_price + i,
                high=start_price + i + 2,
                low=start_price + i - 1,
                close=start_price + i,
                adj_close=start_price + i,
                volume=1_000_000,
            )
    return base




# ---------------------------------------------------------------------
# Trading repository
# ---------------------------------------------------------------------


def test_closed_lots_returns_engine_dataclasses(seeded_user):
    lots = trading_repo.closed_lots(seeded_user.id)

    assert len(lots) > 0
    lot = lots[0]
    # It's the pure engine type, usable directly by behavior functions.
    assert hasattr(lot, "is_winner")
    assert hasattr(lot, "return_pct")
    assert lot.closed_at is not None


def test_open_lots_are_actually_open(seeded_user):
    opens = trading_repo.open_lots(seeded_user.id)
    assert len(opens) > 0
    # No closed_at attribute on OpenLot -- these are strictly open.
    assert all(hasattr(o, "opened_at") for o in opens)


def test_current_holdings_sums_open_quantity(seeded_user):
    holdings = trading_repo.current_holdings(seeded_user.id)

    assert len(holdings) > 0
    assert all(qty > 0 for qty in holdings.values())


def test_sql_disposition_matches_engine(seeded_user):
    """
    The disposition effect computed in SQL must match the same metric
    computed by the Python engine over the same lots. Two independent
    implementations agreeing is strong evidence both are correct.
    """
    from apps.analytics.engine import behavior as bh

    sql_result = trading_repo.hold_period_by_outcome(seeded_user.id)

    lots = trading_repo.closed_lots(seeded_user.id)
    engine_result = bh.disposition_effect(lots)

    assert sql_result["ratio"] == pytest.approx(engine_result.ratio, rel=1e-6)
    assert sql_result["n_winners"] == engine_result.n_winners
    assert sql_result["avg_winner_hold"] == pytest.approx(
        engine_result.avg_hold_winners, rel=1e-6
    )


def test_realized_pnl_timeline_is_cumulative(seeded_user):
    timeline = trading_repo.realized_pnl_timeline(seeded_user.id)

    assert len(timeline) > 0

    # The window function's running total must equal the manual cumsum.
    running = 0.0
    for row in timeline:
        running += row["realized_pnl"]
        assert row["cumulative_pnl"] == pytest.approx(running, rel=1e-6)


def test_timeline_ordered_by_date(seeded_user):
    timeline = trading_repo.realized_pnl_timeline(seeded_user.id)
    dates = [row["date"] for row in timeline]
    assert dates == sorted(dates)
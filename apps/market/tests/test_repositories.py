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
# Market repository
# ---------------------------------------------------------------------


def test_price_frame_pivots_to_wide(price_history):
    frame = market_repo.price_frame(
        ["AAPL", "MSFT"], price_history, price_history + timedelta(days=29)
    )

    assert list(frame.columns) == ["AAPL", "MSFT"] or set(frame.columns) == {"AAPL", "MSFT"}
    assert len(frame) == 30
    # adj_close should come back as float, not Decimal.
    assert frame["AAPL"].dtype == float
    # First AAPL bar was 180.0.
    assert frame["AAPL"].iloc[0] == pytest.approx(180.0)


def test_price_frame_respects_date_bounds(price_history):
    frame = market_repo.price_frame(
        ["AAPL"], price_history, price_history + timedelta(days=9)
    )
    assert len(frame) == 10


def test_price_frame_empty_symbols():
    assert market_repo.price_frame([], date(2026, 1, 1), date(2026, 2, 1)).empty


def test_latest_prices(price_history):
    latest = market_repo.latest_prices(["AAPL", "MSFT"])

    # Last bar is day 29: AAPL 180+29=209, MSFT 400+29=429.
    assert latest["AAPL"] == pytest.approx(209.0)
    assert latest["MSFT"] == pytest.approx(429.0)


def test_coverage(price_history):
    cov = market_repo.coverage(["AAPL"])

    assert cov["AAPL"]["bar_count"] == 30
    assert cov["AAPL"]["first_date"] == price_history
    assert cov["AAPL"]["last_date"] == price_history + timedelta(days=29)

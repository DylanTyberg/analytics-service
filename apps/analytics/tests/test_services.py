"""
Tests for the service layer.

These exercise the full assembly against a real database: seed data,
run an analysis, check the persisted AnalysisRun and metrics. The engine
and repositories are tested in isolation elsewhere; here the concern is
orchestration -- does a run get created, does it succeed, does caching
work, does a failure record a reason instead of vanishing.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest
from django.test import override_settings
from django.utils import timezone

from apps.analytics import services
from apps.analytics.models import AnalysisRun
from apps.market.models import DailyBar, Security
from apps.trading.models import AppUser, PositionLot


@pytest.fixture
def user(db):
    return AppUser.objects.create(cognito_sub="svc-test")


def _make_bars(symbol: str, days: int, start_price: float, drift: float = 0.3):
    sec, _ = Security.objects.get_or_create(symbol=symbol, defaults={"name": symbol})
    base = date(2024, 1, 1)
    price = start_price
    bars = []
    for i in range(days):
        price += drift + (i % 5 - 2) * 0.5   # gentle trend with wiggle
        bars.append(DailyBar(
            security=sec, bar_date=base + timedelta(days=i),
            open=price, high=price + 1, low=price - 1,
            close=price, adj_close=price, volume=1_000_000,
        ))
    DailyBar.objects.bulk_create(bars)
    return sec


def _open_lot(user, sec, qty, price, opened):
    return PositionLot.objects.create(
        user=user, security=sec,
        original_qty=Decimal(str(qty)), remaining_qty=Decimal(str(qty)),
        open_price=Decimal(str(price)), opened_at=opened,
    )


# ---------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------


def test_risk_run_succeeds_and_persists_metrics(user):
    sec_a = _make_bars("AAA", 400, 100.0, drift=0.2)
    sec_b = _make_bars("BBB", 400, 50.0, drift=0.4)
    opened = timezone.now() - timedelta(days=30)
    _open_lot(user, sec_a, 10, 100, opened)
    _open_lot(user, sec_b, 20, 50, opened)

    run = services.get_or_run_risk(user, as_of=date(2025, 1, 1))

    assert run.status == AnalysisRun.Status.SUCCEEDED
    assert run.duration_ms is not None
    assert hasattr(run, "risk_metrics")

    m = run.risk_metrics
    assert m.portfolio_vol > 0
    # Two symbols included, none excluded.
    assert set(run.securities_included) == {"AAA", "BBB"}
    assert run.securities_excluded == {}
    # Correlation matrix persisted in the documented shape.
    assert set(m.correlation_matrix["symbols"]) == {"AAA", "BBB"}


def test_risk_run_fails_cleanly_with_no_positions(user):
    run = services.get_or_run_risk(user, as_of=date(2025, 1, 1))

    assert run.status == AnalysisRun.Status.FAILED
    assert "no open positions" in run.error_message.lower()
    # A failed run is still a record -- not an exception bubbling to the view.
    assert AnalysisRun.objects.filter(pk=run.pk).exists()


def test_risk_run_fails_with_single_position(user):
    """Correlation-based metrics need at least two positions."""
    sec = _make_bars("AAA", 400, 100.0)
    _open_lot(user, sec, 10, 100, timezone.now() - timedelta(days=30))

    run = services.get_or_run_risk(user, as_of=date(2025, 1, 1))

    assert run.status == AnalysisRun.Status.FAILED
    assert "two positions" in run.error_message.lower()


# ---------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------


def test_second_call_returns_cached_run(user):
    sec_a = _make_bars("AAA", 400, 100.0, drift=0.2)
    sec_b = _make_bars("BBB", 400, 50.0, drift=0.4)
    opened = timezone.now() - timedelta(days=30)
    _open_lot(user, sec_a, 10, 100, opened)
    _open_lot(user, sec_b, 20, 50, opened)

    first = services.get_or_run_risk(user, as_of=date(2025, 1, 1))
    second = services.get_or_run_risk(user, as_of=date(2025, 1, 1))

    # Same row, not a recompute.
    assert first.pk == second.pk
    assert AnalysisRun.objects.filter(
        user=user, run_type=AnalysisRun.RunType.RISK
    ).count() == 1


def test_force_bypasses_cache(user):
    sec_a = _make_bars("AAA", 400, 100.0, drift=0.2)
    sec_b = _make_bars("BBB", 400, 50.0, drift=0.4)
    opened = timezone.now() - timedelta(days=30)
    _open_lot(user, sec_a, 10, 100, opened)
    _open_lot(user, sec_b, 20, 50, opened)

    first = services.get_or_run_risk(user, as_of=date(2025, 1, 1))
    second = services.get_or_run_risk(user, as_of=date(2025, 1, 1), force=True)

    assert first.pk != second.pk
    assert AnalysisRun.objects.filter(
        user=user, run_type=AnalysisRun.RunType.RISK
    ).count() == 2


def test_failed_runs_are_not_cached(user):
    """A cache hit must require SUCCEEDED -- a prior failure recomputes."""
    first = services.get_or_run_risk(user, as_of=date(2025, 1, 1))
    assert first.status == AnalysisRun.Status.FAILED

    second = services.get_or_run_risk(user, as_of=date(2025, 1, 1))
    # New attempt, not the cached failure.
    assert first.pk != second.pk


# ---------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------


def test_behavior_run_against_seeded_data(user):
    with override_settings(DEBUG=True):
        from django.core.management import call_command
        call_command("seed_trades", user="svc-test", lots=200,
                     disposition=1.6, seed=5, verbosity=0)

    # Seed prices for every symbol the seeder uses, so buy-hold has inputs
    # for all lots rather than just one.
    from apps.trading.management.commands.seed_trades import UNIVERSE
    for sym, price, _vol in UNIVERSE:
        _make_bars(sym, 50, float(price))

    run = services.get_or_run_behavior(user, as_of=timezone.now().date())

    assert run.status == AnalysisRun.Status.SUCCEEDED
    m = run.behavior_metrics
    assert m.total_closed_lots > 0
    assert m.disposition_ratio is not None
    # Seeded at 1.6, allow sampling slack.
    assert float(m.disposition_ratio) == pytest.approx(1.6, abs=0.5)


def test_behavior_fails_with_no_trades(user):
    run = services.get_or_run_behavior(user, as_of=timezone.now().date())

    assert run.status == AnalysisRun.Status.FAILED
    assert "no closed trades" in run.error_message.lower()
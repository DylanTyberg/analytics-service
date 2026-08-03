"""
Tests for the API layer.

Run against the auth stub (AUTH_STUB=True), so no real Cognito tokens are
needed -- the concern here is routing, serialization, and status handling,
not JWT validation. The stub authenticates every request as a fixed
local-dev user.

The service layer is tested directly elsewhere; these confirm the HTTP
surface wraps it correctly.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest
from django.test import override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from apps.market.models import DailyBar, Security
from apps.trading.models import AppUser, PositionLot


pytestmark = pytest.mark.django_db


@pytest.fixture
def client():
    return APIClient()


@pytest.fixture
def stub_user(db):
    # Matches the sub the auth stub creates, so seeded data attaches to the
    # same user the API authenticates as.
    return AppUser.objects.get_or_create(cognito_sub="local-dev-user")[0]


def _bars(symbol, days, start_price, drift=0.3):
    sec = Security.objects.create(symbol=symbol, name=symbol)
    base = date(2024, 1, 1)
    price = start_price
    rows = []
    for i in range(days):
        price += drift + (i % 5 - 2) * 0.5
        rows.append(DailyBar(
            security=sec, bar_date=base + timedelta(days=i),
            open=price, high=price + 1, low=price - 1,
            close=price, adj_close=price, volume=1_000_000,
        ))
    DailyBar.objects.bulk_create(rows)
    return sec


def _open_lot(user, sec, qty, price):
    PositionLot.objects.create(
        user=user, security=sec,
        original_qty=Decimal(str(qty)), remaining_qty=Decimal(str(qty)),
        open_price=Decimal(str(price)),
        opened_at=timezone.now() - timedelta(days=30),
    )


# ---------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------


@override_settings(AUTH_STUB=False)
def test_unauthenticated_request_is_401(client):
    resp = client.get("/api/v1/analytics/risk")
    assert resp.status_code == 401


def test_health_needs_no_data(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------
# Risk endpoint
# ---------------------------------------------------------------------


def test_risk_endpoint_returns_metrics(client, stub_user):
    a = _bars("AAA", 400, 100.0, drift=0.2)
    b = _bars("BBB", 400, 50.0, drift=0.4)
    _open_lot(stub_user, a, 10, 100)
    _open_lot(stub_user, b, 20, 50)

    resp = client.get("/api/v1/analytics/risk")
    assert resp.status_code == 200

    body = resp.json()
    assert body["run_type"] == "risk"
    assert body["status"] == "succeeded"
    assert body["metrics"]["portfolio_vol"] is not None
    assert set(body["securities_included"]) == {"AAA", "BBB"}


def test_risk_endpoint_reports_failure_as_200(client, stub_user):
    """
    No positions -> the analysis fails, but that is a valid result the
    client should read, not an HTTP error.
    """
    resp = client.get("/api/v1/analytics/risk")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["metrics"] is None
    assert "no open positions" in body["error_message"].lower()


def test_risk_force_recomputes(client, stub_user):
    a = _bars("AAA", 400, 100.0, drift=0.2)
    b = _bars("BBB", 400, 50.0, drift=0.4)
    _open_lot(stub_user, a, 10, 100)
    _open_lot(stub_user, b, 20, 50)

    first = client.get("/api/v1/analytics/risk").json()
    cached = client.get("/api/v1/analytics/risk").json()
    forced = client.get("/api/v1/analytics/risk?force=true").json()

    assert first["id"] == cached["id"]       # served from cache
    assert forced["id"] != first["id"]       # recomputed


# ---------------------------------------------------------------------
# Behaviour endpoint
# ---------------------------------------------------------------------


def test_behavior_endpoint(client, stub_user):
    with override_settings(DEBUG=True):
        from django.core.management import call_command
        call_command("seed_trades", user="local-dev-user", lots=150,
                     disposition=1.6, seed=3, verbosity=0)

    resp = client.get("/api/v1/analytics/behavior")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "succeeded"
    assert body["run_type"] == "behavior"
    assert body["metrics"]["disposition_ratio"] is not None
    assert body["metrics"]["total_closed_lots"] > 0


def test_behavior_no_trades_is_failed_200(client, stub_user):
    resp = client.get("/api/v1/analytics/behavior")

    assert resp.status_code == 200
    assert resp.json()["status"] == "failed"
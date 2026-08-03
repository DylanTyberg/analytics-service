"""
Tests for Polygon ingestion.

The HTTP layer is mocked -- these must never hit the real API (network
flakiness, and no reason to spend quota). The client's parsing and the
command's upsert/idempotency logic are what matter, and both are testable
without a real request.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest
from django.core.management import call_command

from apps.market.models import DailyBar, Security
from apps.market.polygon_client import Bar, PolygonClient, PolygonError


def _epoch_ms(y, m, d) -> int:
    return int(datetime(y, m, d, 14, 30, tzinfo=timezone.utc).timestamp() * 1000)


def _polygon_payload(rows, status="OK"):
    return {
        "status": status,
        "results": [
            {"t": _epoch_ms(*r["date"]), "o": r["o"], "h": r["h"],
             "l": r["l"], "c": r["c"], "v": r["v"]}
            for r in rows
        ],
    }


def _mock_response(payload, status_code=200):
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = ""
    return resp


# ---------------------------------------------------------------------
# Client parsing
# ---------------------------------------------------------------------


def test_client_parses_bars():
    payload = _polygon_payload([
        {"date": (2026, 1, 2), "o": 100.1, "h": 101.5, "l": 99.8, "c": 100.9, "v": 5000000},
        {"date": (2026, 1, 3), "o": 100.9, "h": 102.0, "l": 100.5, "c": 101.7, "v": 4200000},
    ])
    session = Mock()
    session.get.return_value = _mock_response(payload)

    client = PolygonClient("fake-key", session=session)
    bars = client.daily_bars("AAPL", date(2026, 1, 1), date(2026, 1, 5))

    assert len(bars) == 2
    assert bars[0].bar_date == date(2026, 1, 2)
    assert bars[0].close == Decimal("100.9000")
    assert bars[0].adj_close == Decimal("100.9000")
    assert bars[0].volume == 5000000


def test_client_requires_key():
    with pytest.raises(PolygonError):
        PolygonClient("")


def test_client_raises_on_rate_limit():
    session = Mock()
    session.get.return_value = _mock_response({}, status_code=429)
    client = PolygonClient("k", session=session)

    with pytest.raises(PolygonError, match="rate limited"):
        client.daily_bars("AAPL", date(2026, 1, 1), date(2026, 1, 5))


def test_client_raises_on_bad_status():
    session = Mock()
    session.get.return_value = _mock_response({"status": "ERROR"})
    client = PolygonClient("k", session=session)

    with pytest.raises(PolygonError):
        client.daily_bars("AAPL", date(2026, 1, 1), date(2026, 1, 5))


def test_client_empty_results():
    session = Mock()
    session.get.return_value = _mock_response({"status": "OK", "results": []})
    client = PolygonClient("k", session=session)

    assert client.daily_bars("AAPL", date(2026, 1, 1), date(2026, 1, 5)) == []


# ---------------------------------------------------------------------
# Command ingest and idempotency
# ---------------------------------------------------------------------


@pytest.fixture
def fake_bars():
    return [
        Bar(date(2026, 1, 2), Decimal("100"), Decimal("101"), Decimal("99"),
            Decimal("100.5"), Decimal("100.5"), 1000000),
        Bar(date(2026, 1, 3), Decimal("100.5"), Decimal("102"), Decimal("100"),
            Decimal("101.5"), Decimal("101.5"), 1100000),
    ]


@pytest.mark.django_db
def test_ingest_creates_bars(fake_bars):
    with patch.object(PolygonClient, "daily_bars", return_value=fake_bars):
        call_command("ingest_bars", symbols=["AAPL"], verbosity=0)

    assert DailyBar.objects.filter(security__symbol="AAPL").count() == 2
    sec = Security.objects.get(symbol="AAPL")
    assert sec.first_bar_date == date(2026, 1, 2)
    assert sec.last_bar_date == date(2026, 1, 3)


@pytest.mark.django_db
def test_ingest_is_idempotent(fake_bars):
    """Re-running upserts on (security, bar_date) -- no duplicates."""
    with patch.object(PolygonClient, "daily_bars", return_value=fake_bars):
        call_command("ingest_bars", symbols=["AAPL"], verbosity=0)
        call_command("ingest_bars", symbols=["AAPL"], verbosity=0)

    assert DailyBar.objects.filter(security__symbol="AAPL").count() == 2


@pytest.mark.django_db
def test_ingest_updates_on_reingest():
    """A re-fetch with re-adjusted prices overwrites, not duplicates."""
    original = [Bar(date(2026, 1, 2), Decimal("100"), Decimal("101"),
                    Decimal("99"), Decimal("100"), Decimal("100"), 1000000)]
    adjusted = [Bar(date(2026, 1, 2), Decimal("50"), Decimal("50.5"),
                    Decimal("49.5"), Decimal("50"), Decimal("50"), 2000000)]  # post-split

    with patch.object(PolygonClient, "daily_bars", return_value=original):
        call_command("ingest_bars", symbols=["AAPL"], verbosity=0)
    with patch.object(PolygonClient, "daily_bars", return_value=adjusted):
        call_command("ingest_bars", symbols=["AAPL"], verbosity=0)

    bar = DailyBar.objects.get(security__symbol="AAPL", bar_date=date(2026, 1, 2))
    assert bar.adj_close == Decimal("50.0000")   # updated, not the original 100
    assert DailyBar.objects.count() == 1


@pytest.mark.django_db
def test_one_bad_symbol_does_not_abort_run(fake_bars):
    """A PolygonError on one symbol is logged; others still ingest."""
    def side_effect(self, symbol, start, end):
        if symbol == "BAD":
            raise PolygonError("boom")
        return fake_bars

    with patch.object(PolygonClient, "daily_bars", side_effect=side_effect, autospec=True):
        call_command("ingest_bars", symbols=["AAPL", "BAD", "MSFT"], verbosity=0)

    assert DailyBar.objects.filter(security__symbol="AAPL").exists()
    assert DailyBar.objects.filter(security__symbol="MSFT").exists()
    assert not DailyBar.objects.filter(security__symbol="BAD").exists()


@pytest.mark.django_db
def test_dry_run_writes_nothing(fake_bars):
    with patch.object(PolygonClient, "daily_bars", return_value=fake_bars):
        call_command("ingest_bars", symbols=["AAPL"], dry_run=True, verbosity=0)

    assert DailyBar.objects.count() == 0


@pytest.mark.django_db
def test_discovers_symbols_from_trades(fake_bars):
    """With no --symbols, the command ingests what the user has traded."""
    from apps.trading.models import AppUser, TradeFill
    from django.utils import timezone

    user = AppUser.objects.create(cognito_sub="disc")
    sec = Security.objects.create(symbol="TSLA", name="TSLA")
    TradeFill.objects.create(
        user=user, security=sec, side="buy",
        quantity=Decimal("1"), price=Decimal("200"),
        filled_at=timezone.now(), source_event_id="e1",
    )

    captured = []

    def capture(self, symbol, start, end):
        captured.append(symbol)
        return fake_bars

    with patch.object(PolygonClient, "daily_bars", side_effect=capture, autospec=True):
        call_command("ingest_bars", verbosity=0)

    assert "TSLA" in captured
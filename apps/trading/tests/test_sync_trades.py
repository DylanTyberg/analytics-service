"""
Tests for sync_trades.

DynamoDB is faked -- a fake reader yields TradeItem objects, so no AWS and
no boto3 mocking gymnastics. The concerns under test are: idempotent
ingest, correct lot rebuild, unknown-symbol skipping, and single-user
scoping.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest
from django.core.management import call_command

from apps.market.models import Security
from apps.trading.dynamo_reader import TradeItem
from apps.trading.models import AppUser, PositionLot, TradeFill

BASE = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)


def trade(tid, sub, symbol, side, qty, price, day=0):
    return TradeItem(
        user_sub=sub, trade_id=tid, symbol=symbol, side=side,
        quantity=Decimal(str(qty)), price=Decimal(str(price)),
        executed_at=BASE + timedelta(days=day),
    )


class FakeReader:
    """Stands in for TradeReader. all_trades() yields canned items."""

    def __init__(self, items):
        self._items = items

    def all_trades(self):
        yield from self._items


def run_sync(items, **opts):
    fake = FakeReader(items)
    with patch("apps.trading.management.commands.sync_trades.TradeReader",
               return_value=fake):
        call_command("sync_trades", table="fake-table", verbosity=0, **opts)


@pytest.fixture
def securities(db):
    for sym in ("AAPL", "MSFT"):
        Security.objects.create(symbol=sym, name=sym)


# ---------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------


@pytest.mark.django_db
def test_ingest_creates_fills(securities):
    run_sync([
        trade("t1", "user-a", "AAPL", "buy", 10, 100, day=0),
        trade("t2", "user-a", "AAPL", "sell", 10, 110, day=10),
    ])

    assert TradeFill.objects.count() == 2
    assert AppUser.objects.filter(cognito_sub="user-a").exists()


@pytest.mark.django_db
def test_ingest_is_idempotent(securities):
    items = [
        trade("t1", "user-a", "AAPL", "buy", 10, 100, day=0),
        trade("t2", "user-a", "AAPL", "sell", 10, 110, day=10),
    ]
    run_sync(items)
    run_sync(items)  # again

    # source_event_id uniqueness prevents duplicates.
    assert TradeFill.objects.count() == 2


@pytest.mark.django_db
def test_unknown_symbol_is_skipped(securities):
    run_sync([
        trade("t1", "user-a", "AAPL", "buy", 10, 100),
        trade("t2", "user-a", "ZZZZ", "buy", 5, 50),   # no such security
    ])

    assert TradeFill.objects.count() == 1
    assert TradeFill.objects.filter(security__symbol="AAPL").exists()


@pytest.mark.django_db
def test_user_scope(securities):
    run_sync(
        [
            trade("t1", "user-a", "AAPL", "buy", 10, 100),
            trade("t2", "user-b", "AAPL", "buy", 5, 100),
        ],
        user="user-a",
    )

    assert TradeFill.objects.count() == 1
    assert not AppUser.objects.filter(cognito_sub="user-b").exists()


# ---------------------------------------------------------------------
# Rematch -- the reason the sync exists
# ---------------------------------------------------------------------


@pytest.mark.django_db
def test_sync_builds_closed_lot(securities):
    run_sync([
        trade("t1", "user-a", "AAPL", "buy", 10, 100, day=0),
        trade("t2", "user-a", "AAPL", "sell", 10, 110, day=10),
    ])

    lots = PositionLot.objects.filter(user__cognito_sub="user-a")
    assert lots.count() == 1

    lot = lots.first()
    assert lot.closed_at is not None
    assert lot.realized_pnl == Decimal("100")   # 10 * (110 - 100)
    assert lot.hold_days == 10


@pytest.mark.django_db
def test_sync_builds_open_lot(securities):
    run_sync([trade("t1", "user-a", "AAPL", "buy", 10, 100)])

    lot = PositionLot.objects.get(user__cognito_sub="user-a")
    assert lot.closed_at is None
    assert lot.remaining_qty == Decimal("10")


@pytest.mark.django_db
def test_incremental_sync_rebuilds_correctly(securities):
    """
    First sync opens a lot; a later sync adds the closing sell. The rebuild
    must produce one correctly closed lot, matching what a single sync of
    both fills would give.
    """
    run_sync([trade("t1", "user-a", "AAPL", "buy", 10, 100, day=0)])
    assert PositionLot.objects.get(user__cognito_sub="user-a").closed_at is None

    run_sync([
        trade("t1", "user-a", "AAPL", "buy", 10, 100, day=0),      # already synced
        trade("t2", "user-a", "AAPL", "sell", 10, 110, day=10),    # new
    ])

    lots = PositionLot.objects.filter(user__cognito_sub="user-a")
    assert lots.count() == 1
    assert lots.first().realized_pnl == Decimal("100")


@pytest.mark.django_db
def test_rematch_only_skips_dynamo(securities):
    """--rematch-only rebuilds lots from existing fills without a reader."""
    user = AppUser.objects.create(cognito_sub="user-a")
    sec = Security.objects.get(symbol="AAPL")
    TradeFill.objects.create(
        user=user, security=sec, side="buy",
        quantity=Decimal("10"), price=Decimal("100"),
        filled_at=BASE, source_event_id="t1",
    )

    # No TradeReader patch: if it tried to hit DynamoDB this would fail.
    call_command("sync_trades", rematch_only=True, verbosity=0)

    assert PositionLot.objects.filter(user=user).count() == 1
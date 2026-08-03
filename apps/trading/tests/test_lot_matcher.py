"""
Tests for FIFO lot matching.

Pure unit tests -- no database, no Django. Every expected value is worked
out by hand in the docstring, so a failure tells you the logic is wrong
rather than merely different from last time.

The cases that matter most are the ones that are easy to get subtly
wrong: a sell spanning several lots, a partial sell leaving a lot open,
and out-of-order input.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from apps.trading.lot_matcher import (
    Allocation,
    Fill,
    Side,
    match_fills,
    summarize_by_symbol,
)

BASE = datetime(2026, 1, 5, 14, 30)


def fill(fid, side, qty, price, day_offset=0, symbol="AAPL"):
    return Fill(
        fill_id=fid,
        symbol=symbol,
        side=Side.BUY if side == "B" else Side.SELL,
        quantity=Decimal(str(qty)),
        price=Decimal(str(price)),
        executed_at=BASE + timedelta(days=day_offset),
    )


# ---------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------


def test_single_buy_opens_one_open_lot():
    result = match_fills([fill("f1", "B", 100, 50)])

    assert len(result.lots) == 1
    lot = result.lots[0]
    assert lot.is_open
    assert lot.remaining_qty == Decimal("100")
    assert lot.closed_at is None
    assert result.closed_lots == []


def test_buy_then_full_sell_closes_lot():
    """Buy 100 @ 50, sell 100 @ 60 ten days later -> +1000 over 10 days."""
    result = match_fills([
        fill("f1", "B", 100, 50),
        fill("f2", "S", 100, 60, day_offset=10),
    ])

    assert len(result.lots) == 1
    lot = result.lots[0]

    assert lot.is_closed
    assert lot.realized_pnl == Decimal("1000")
    assert lot.hold_days == 10
    assert lot.close_price == Decimal("60")
    assert lot.fill_ids == ["f1", "f2"]


def test_loss_is_negative():
    result = match_fills([
        fill("f1", "B", 10, 100),
        fill("f2", "S", 10, 80, day_offset=3),
    ])

    assert result.lots[0].realized_pnl == Decimal("-200")


# ---------------------------------------------------------------------
# Partial fills
# ---------------------------------------------------------------------


def test_partial_sell_leaves_lot_open():
    """
    Buy 100, sell 30. The lot stays open with 70 remaining, and the
    realised P&L so far covers only the 30 sold.
    """
    result = match_fills([
        fill("f1", "B", 100, 50),
        fill("f2", "S", 30, 60, day_offset=5),
    ])

    lot = result.lots[0]
    assert lot.is_open
    assert lot.remaining_qty == Decimal("70")
    assert lot.realized_pnl == Decimal("300")  # 30 * (60 - 50)
    assert lot.closed_at is None
    assert lot.hold_days is None

    # The allocation records the partial sale.
    assert len(result.allocations) == 1
    assert result.allocations[0].quantity == Decimal("30")


def test_lot_closed_by_two_separate_sells():
    """
    Buy 100 @ 50. Sell 40 @ 60, then 60 @ 55.
    P&L = 40*10 + 60*5 = 400 + 300 = 700.
    hold_days measured to the FINAL sell, which closed it.
    """
    result = match_fills([
        fill("f1", "B", 100, 50),
        fill("f2", "S", 40, 60, day_offset=5),
        fill("f3", "S", 60, 55, day_offset=12),
    ])

    lot = result.lots[0]
    assert lot.is_closed
    assert lot.realized_pnl == Decimal("700")
    assert lot.hold_days == 12
    assert lot.close_price == Decimal("55")
    assert len(result.allocations) == 2


# ---------------------------------------------------------------------
# FIFO ordering -- the core behaviour
# ---------------------------------------------------------------------


def test_sell_consumes_oldest_lot_first():
    """
    Two buys: 100 @ 50 (day 0), 100 @ 70 (day 5). Sell 100 @ 80 on day 10.
    FIFO consumes the OLDEST, so P&L must be 100*(80-50) = 3000,
    not 100*(80-70) = 1000.
    """
    result = match_fills([
        fill("f1", "B", 100, 50),
        fill("f2", "B", 100, 70, day_offset=5),
        fill("f3", "S", 100, 80, day_offset=10),
    ])

    closed = result.closed_lots
    assert len(closed) == 1
    assert closed[0].open_price == Decimal("50")
    assert closed[0].realized_pnl == Decimal("3000")

    # The newer lot is untouched.
    open_lots = result.open_lots
    assert len(open_lots) == 1
    assert open_lots[0].open_price == Decimal("70")
    assert open_lots[0].remaining_qty == Decimal("100")


def test_sell_spanning_multiple_lots():
    """
    Buys: 50 @ 10 (day 0), 50 @ 20 (day 2), 50 @ 30 (day 4).
    Sell 120 @ 40 on day 10 -- consumes lot 1 fully, lot 2 fully,
    and 20 of lot 3.

    P&L = 50*(40-10) + 50*(40-20) + 20*(40-30)
        = 1500 + 1000 + 200 = 2700
    """
    result = match_fills([
        fill("f1", "B", 50, 10),
        fill("f2", "B", 50, 20, day_offset=2),
        fill("f3", "B", 50, 30, day_offset=4),
        fill("f4", "S", 120, 40, day_offset=10),
    ])

    assert len(result.closed_lots) == 2
    assert len(result.open_lots) == 1

    total_pnl = sum(lot.realized_pnl for lot in result.lots)
    assert total_pnl == Decimal("2700")

    # Three allocations from one sell fill.
    assert len(result.allocations) == 3
    assert all(a.sell_fill_id == "f4" for a in result.allocations)

    remaining = result.open_lots[0]
    assert remaining.open_price == Decimal("30")
    assert remaining.remaining_qty == Decimal("30")


def test_hold_days_differ_per_lot_in_multi_lot_sell():
    """
    Each lot's hold period is measured from ITS OWN purchase date, not
    from the first. This is the whole reason FIFO lots exist rather than
    an average-cost position.
    """
    result = match_fills([
        fill("f1", "B", 10, 100),                 # day 0
        fill("f2", "B", 10, 100, day_offset=30),  # day 30
        fill("f3", "S", 20, 110, day_offset=60),  # day 60
    ])

    holds = sorted(lot.hold_days for lot in result.closed_lots)
    assert holds == [30, 60]


# ---------------------------------------------------------------------
# Symbol isolation
# ---------------------------------------------------------------------


def test_symbols_are_matched_independently():
    """Selling MSFT must never consume an AAPL lot."""
    result = match_fills([
        fill("f1", "B", 100, 50, symbol="AAPL"),
        fill("f2", "B", 100, 200, symbol="MSFT", day_offset=1),
        fill("f3", "S", 100, 60, symbol="AAPL", day_offset=5),
    ])

    closed = result.closed_lots
    assert len(closed) == 1
    assert closed[0].symbol == "AAPL"
    assert closed[0].realized_pnl == Decimal("1000")

    open_lots = result.open_lots
    assert len(open_lots) == 1
    assert open_lots[0].symbol == "MSFT"


# ---------------------------------------------------------------------
# Ordering and determinism
# ---------------------------------------------------------------------


def test_input_is_sorted_before_matching():
    """
    Fills arriving out of order (DynamoDB pagination gives no cross-page
    ordering guarantee) must still match chronologically.
    """
    unordered = [
        fill("f3", "S", 100, 80, day_offset=10),
        fill("f1", "B", 100, 50, day_offset=0),
        fill("f2", "B", 100, 70, day_offset=5),
    ]

    result = match_fills(unordered)

    assert len(result.closed_lots) == 1
    assert result.closed_lots[0].open_price == Decimal("50")
    assert result.unmatched_sells == {}


def test_same_timestamp_is_deterministic():
    """
    Two fills at the identical timestamp are tie-broken by fill_id, so
    repeated runs over the same input always produce the same lots.
    """
    fills = [
        Fill("b_second", "AAPL", Side.BUY, Decimal("10"), Decimal("20"), BASE),
        Fill("a_first", "AAPL", Side.BUY, Decimal("10"), Decimal("10"), BASE),
        fill("z_sell", "S", 10, 30, day_offset=1),
    ]

    first = match_fills(fills)
    second = match_fills(list(reversed(fills)))

    assert first.closed_lots[0].open_price == second.closed_lots[0].open_price
    # "a_first" sorts before "b_second", so the 10-priced lot is consumed.
    assert first.closed_lots[0].open_price == Decimal("10")


# ---------------------------------------------------------------------
# Degenerate and defensive cases
# ---------------------------------------------------------------------


def test_oversell_is_recorded_not_raised():
    """
    Selling more than was bought indicates a gap in the fill history.
    One bad symbol must not abort the whole matching run, so it is
    recorded and matching continues.
    """
    result = match_fills([
        fill("f1", "B", 50, 10),
        fill("f2", "S", 80, 20, day_offset=1),
    ])

    assert result.unmatched_sells == {"f2": Decimal("30")}
    # The 50 that DID exist still matched correctly.
    assert len(result.closed_lots) == 1
    assert result.closed_lots[0].realized_pnl == Decimal("500")


def test_sell_with_no_open_lots():
    result = match_fills([fill("f1", "S", 10, 50)])

    assert result.lots == []
    assert result.unmatched_sells == {"f1": Decimal("10")}


def test_empty_input():
    result = match_fills([])

    assert result.lots == []
    assert result.allocations == []
    assert result.unmatched_sells == {}


def test_same_day_round_trip_is_zero_hold_days():
    """Day trading: bought and sold the same day."""
    result = match_fills([
        Fill("f1", "AAPL", Side.BUY, Decimal("10"), Decimal("50"), BASE),
        Fill("f2", "AAPL", Side.SELL, Decimal("10"), Decimal("52"),
             BASE + timedelta(hours=2)),
    ])

    assert result.closed_lots[0].hold_days == 0
    assert result.closed_lots[0].realized_pnl == Decimal("20")


def test_fractional_shares():
    result = match_fills([
        fill("f1", "B", "0.5", 100),
        fill("f2", "S", "0.25", 120, day_offset=2),
    ])

    lot = result.lots[0]
    assert lot.remaining_qty == Decimal("0.25")
    assert lot.realized_pnl == Decimal("5.00")  # 0.25 * 20


def test_dust_residue_closes_lot():
    """
    Repeated partial sells can leave a sliver far below any meaningful
    quantity. It should close the lot rather than leave it open forever.
    """
    result = match_fills([
        fill("f1", "B", "1.0000000", 100),
        fill("f2", "S", "0.9999999", 110, day_offset=1),
    ])

    assert result.lots[0].is_closed


def test_rejects_invalid_fill():
    with pytest.raises(ValueError):
        Fill("bad", "AAPL", Side.BUY, Decimal("0"), Decimal("10"), BASE)

    with pytest.raises(ValueError):
        Fill("bad", "AAPL", Side.BUY, Decimal("-5"), Decimal("10"), BASE)


# ---------------------------------------------------------------------
# Incremental matching
# ---------------------------------------------------------------------


def test_incremental_run_continues_from_existing_lots():
    """
    The ETL matches only new fills against already-open lots rather than
    rebuilding from the entire history every run.
    """
    first = match_fills([fill("f1", "B", 100, 50)])
    assert len(first.open_lots) == 1

    second = match_fills(
        [fill("f2", "S", 100, 60, day_offset=10)],
        existing_lots=first.lots,
        starting_lot_id=99,
    )

    assert len(second.closed_lots) == 1
    assert second.closed_lots[0].realized_pnl == Decimal("1000")
    assert second.closed_lots[0].hold_days == 10


def test_incremental_matches_full_rebuild():
    """
    Two increments must produce the same result as one rebuild over the
    complete history -- otherwise the ETL and a --rebuild would disagree.
    """
    all_fills = [
        fill("f1", "B", 50, 10),
        fill("f2", "B", 50, 20, day_offset=2),
        fill("f3", "S", 70, 30, day_offset=5),
    ]

    rebuild = match_fills(all_fills)

    step1 = match_fills(all_fills[:2])
    step2 = match_fills(all_fills[2:], existing_lots=step1.lots, starting_lot_id=3)

    assert (
        sum(l.realized_pnl for l in step2.lots)
        == sum(l.realized_pnl for l in rebuild.lots)
    )
    assert len(step2.closed_lots) == len(rebuild.closed_lots)


# ---------------------------------------------------------------------
# Allocation helpers
# ---------------------------------------------------------------------


def test_allocation_return_pct():
    alloc = Allocation(
        sell_fill_id="f2",
        lot_id=1,
        quantity=Decimal("10"),
        open_price=Decimal("100"),
        close_price=Decimal("125"),
        opened_at=BASE,
        closed_at=BASE + timedelta(days=7),
    )

    assert alloc.return_pct == Decimal("0.25")
    assert alloc.realized_pnl == Decimal("250")
    assert alloc.hold_days == 7


def test_summarize_by_symbol():
    result = match_fills([
        fill("f1", "B", 100, 50, symbol="AAPL"),
        fill("f2", "S", 100, 60, day_offset=10, symbol="AAPL"),
        fill("f3", "B", 10, 200, symbol="MSFT"),
    ])

    summary = summarize_by_symbol(result)

    assert summary["AAPL"]["closed_lots"] == 1
    assert summary["AAPL"]["realized_pnl"] == Decimal("1000")
    assert summary["AAPL"]["avg_hold_days"] == 10
    assert summary["MSFT"]["open_qty"] == Decimal("10")
    assert summary["MSFT"]["closed_lots"] == 0
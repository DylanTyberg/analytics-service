"""
FIFO lot matching.

Turns an ordered stream of fills into position lots: each buy opens a lot,
each sell consumes the oldest open lot(s) first. This is the step that
makes the behavioural analytics tractable -- once lots exist with
realised_pnl and hold_days on them, every metric is a simple aggregate
instead of a self-join over the raw fill log.

The core here is pure: it takes plain dataclasses and returns plain
dataclasses, with no Django imports and no database access. The Django
wrapper that persists the results lives in services.py. That split is
what lets the tricky cases -- partial fills, sells spanning several lots,
overselling -- be tested exhaustively in milliseconds.

FIFO is chosen over average-cost or LIFO because it preserves the
identity of individual purchases, which is what the behavioural metrics
actually need: "how long was THIS specific purchase held, and did it make
money?" Average-cost blends everything into one position and destroys
exactly that information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum

# Quantities below this are treated as zero. Floating point does not
# reach here (everything is Decimal), but repeated partial fills can
# leave a residue like 1E-9 that should close the lot rather than leave
# it open forever with a meaningless sliver.
DUST = Decimal("0.000001")


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Fill:
    """One executed trade, as read from the trade log."""

    fill_id: str
    symbol: str
    side: Side
    quantity: Decimal
    price: Decimal
    executed_at: datetime

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"fill {self.fill_id}: quantity must be positive")
        if self.price < 0:
            raise ValueError(f"fill {self.fill_id}: price must be non-negative")


@dataclass
class Lot:
    """
    A single purchase, tracked from open to close.

    `remaining_qty` is decremented as sells consume it. The lot closes
    when it reaches zero, at which point closed_at, close_price and
    hold_days are set.
    """

    lot_id: int
    symbol: str
    original_qty: Decimal
    remaining_qty: Decimal
    open_price: Decimal
    opened_at: datetime

    closed_at: datetime | None = None
    close_price: Decimal | None = None
    realized_pnl: Decimal = Decimal("0")
    hold_days: int | None = None

    # Every fill that touched this lot: the opening buy and each sell
    # that consumed part of it.
    fill_ids: list[str] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.remaining_qty > DUST

    @property
    def is_closed(self) -> bool:
        return not self.is_open


@dataclass(frozen=True)
class Allocation:
    """
    One sell fill consuming one lot.

    A single sell can span several lots, producing several allocations.
    Kept separately from the lots because it is the audit trail: it
    records exactly which purchase each sale was matched against.
    """

    sell_fill_id: str
    lot_id: int
    quantity: Decimal
    open_price: Decimal
    close_price: Decimal
    opened_at: datetime
    closed_at: datetime

    @property
    def realized_pnl(self) -> Decimal:
        return (self.close_price - self.open_price) * self.quantity

    @property
    def hold_days(self) -> int:
        return (self.closed_at - self.opened_at).days

    @property
    def return_pct(self) -> Decimal:
        if self.open_price == 0:
            return Decimal("0")
        return (self.close_price - self.open_price) / self.open_price


@dataclass
class MatchResult:
    lots: list[Lot] = field(default_factory=list)
    allocations: list[Allocation] = field(default_factory=list)
    # Sells that could not be fully matched against open lots, as
    # {fill_id: unmatched_quantity}. See the note in match_fills.
    unmatched_sells: dict[str, Decimal] = field(default_factory=dict)

    @property
    def closed_lots(self) -> list[Lot]:
        return [lot for lot in self.lots if lot.is_closed]

    @property
    def open_lots(self) -> list[Lot]:
        return [lot for lot in self.lots if lot.is_open]


def match_fills(
    fills: list[Fill],
    *,
    starting_lot_id: int = 1,
    existing_lots: list[Lot] | None = None,
) -> MatchResult:
    """
    Match an ordered stream of fills into lots.

    Fills are sorted by execution time before processing -- FIFO is
    meaningless if the input order is arbitrary, and DynamoDB pagination
    does not guarantee ordering across pages.

    `existing_lots` supports incremental runs: pass the currently open
    lots and only the new fills, and matching continues where it left
    off. Omit it for a full rebuild from the complete fill history.

    Symbols are matched independently -- selling AAPL never touches an
    MSFT lot -- which the per-symbol lot queues below enforce.
    """
    result = MatchResult()
    next_lot_id = starting_lot_id

    # Open lots per symbol, oldest first. Using a list rather than a
    # deque because partial consumption mutates the head in place far
    # more often than it pops it.
    open_by_symbol: dict[str, list[Lot]] = {}

    if existing_lots:
        for lot in existing_lots:
            result.lots.append(lot)
            if lot.is_open:
                open_by_symbol.setdefault(lot.symbol, []).append(lot)
            next_lot_id = max(next_lot_id, lot.lot_id + 1)

        for queue in open_by_symbol.values():
            queue.sort(key=lambda l: (l.opened_at, l.lot_id))

    # Sort by time, then fill_id as a deterministic tiebreak. Two fills
    # can share a timestamp to the millisecond; without the secondary key
    # the match order would vary between runs and produce different lots
    # from identical input.
    ordered = sorted(fills, key=lambda f: (f.executed_at, f.fill_id))

    for fill in ordered:
        queue = open_by_symbol.setdefault(fill.symbol, [])

        if fill.side is Side.BUY:
            lot = Lot(
                lot_id=next_lot_id,
                symbol=fill.symbol,
                original_qty=fill.quantity,
                remaining_qty=fill.quantity,
                open_price=fill.price,
                opened_at=fill.executed_at,
                fill_ids=[fill.fill_id],
            )
            next_lot_id += 1
            result.lots.append(lot)
            queue.append(lot)
            continue

        # --- SELL: consume oldest open lots first ---
        to_sell = fill.quantity

        while to_sell > DUST and queue:
            lot = queue[0]
            take = min(to_sell, lot.remaining_qty)

            result.allocations.append(
                Allocation(
                    sell_fill_id=fill.fill_id,
                    lot_id=lot.lot_id,
                    quantity=take,
                    open_price=lot.open_price,
                    close_price=fill.price,
                    opened_at=lot.opened_at,
                    closed_at=fill.executed_at,
                )
            )

            lot.remaining_qty -= take
            lot.realized_pnl += (fill.price - lot.open_price) * take
            lot.fill_ids.append(fill.fill_id)
            to_sell -= take

            if not lot.is_open:
                # Fully consumed. close_price is the price of the sell
                # that finished it off; for a lot closed by several sells
                # this is the last one, while realized_pnl correctly
                # accumulates across all of them.
                lot.remaining_qty = Decimal("0")
                lot.closed_at = fill.executed_at
                lot.close_price = fill.price
                lot.hold_days = (fill.executed_at - lot.opened_at).days
                queue.pop(0)

        if to_sell > DUST:
            # Sold more than was ever bought. Should be impossible -- the
            # trade endpoint rejects oversells -- but a gap in the fill
            # history (a sync that missed a buy) produces it too. Record
            # it rather than raising: one bad symbol must not abort the
            # entire user's matching run.
            result.unmatched_sells[fill.fill_id] = to_sell

    return result


def summarize_by_symbol(result: MatchResult) -> dict[str, dict]:
    """
    Per-symbol rollup. Useful for spot-checking a matching run and for
    the position-level breakdown in the UI.
    """
    summary: dict[str, dict] = {}

    for lot in result.lots:
        s = summary.setdefault(
            lot.symbol,
            {
                "open_qty": Decimal("0"),
                "closed_lots": 0,
                "realized_pnl": Decimal("0"),
                "total_hold_days": 0,
            },
        )

        if lot.is_open:
            s["open_qty"] += lot.remaining_qty
        else:
            s["closed_lots"] += 1
            s["realized_pnl"] += lot.realized_pnl
            s["total_hold_days"] += lot.hold_days or 0

    for s in summary.values():
        s["avg_hold_days"] = (
            s["total_hold_days"] / s["closed_lots"] if s["closed_lots"] else None
        )

    return summary
"""
Seed synthetic trade history for local development and tests.

This exists for one reason: you cannot validate an analytics engine
against data whose true answer you do not know. So this generates
populations with KNOWN properties -- a target disposition ratio, a target
win rate -- which the engine tests then assert against.

Design note on lot construction
-------------------------------
An earlier version routed generated fills through the real FIFO matcher.
That destroyed the signal: FIFO closes the OLDEST open lot for a symbol,
not the buy a sell was "meant" to pair with, so intended hold periods got
scrambled and winner/loser holds converged. FIFO was behaving correctly --
the mismatch was between paired generation and oldest-first matching.

So the seeder now builds PositionLot rows DIRECTLY from the pairs it
generated, where it knows the true hold and P&L. TradeFill rows are still
written for realism, but they are not the source of the lots. The matcher
is tested separately in test_lot_matcher.py with hand-built fills, which
is the right place for it.

Guarded against production: refuses to run when DEBUG is False.

Usage:
    python manage.py seed_trades --user demo --lots 200
    python manage.py seed_trades --user demo --lots 200 --disposition 1.8 --seed 42
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from apps.market.models import Security
from apps.trading.models import AppUser, PositionLot, TradeFill

UNIVERSE = [
    ("AAPL", Decimal("180"), 0.28),
    ("MSFT", Decimal("410"), 0.26),
    ("NVDA", Decimal("880"), 0.52),
    ("AMZN", Decimal("175"), 0.34),
    ("GOOGL", Decimal("155"), 0.30),
    ("JPM", Decimal("195"), 0.24),
    ("XLF", Decimal("40"), 0.18),
    ("TSLA", Decimal("240"), 0.58),
    ("KO", Decimal("62"), 0.15),
    ("PG", Decimal("165"), 0.16),
]


@dataclass
class SeedConfig:
    n_lots: int = 200
    win_probability: float = 0.52
    disposition_ratio: float = 1.6
    base_winner_hold_days: int = 12
    open_fraction: float = 0.20
    history_days: int = 540


@dataclass
class GeneratedLot:
    """
    A fully-specified lot with its known outcome. This is the intended
    truth; the persisted PositionLot mirrors it exactly.
    """

    symbol: str
    quantity: Decimal
    open_price: Decimal
    opened_at: object          # datetime
    close_price: Decimal | None
    closed_at: object | None
    realized_pnl: Decimal | None
    hold_days: int | None

    @property
    def is_closed(self) -> bool:
        return self.closed_at is not None


class Command(BaseCommand):
    help = "Seed synthetic trade history for development and testing."
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sec_cache: dict[str, int] = {}

    def add_arguments(self, parser):
        parser.add_argument("--user", required=True)
        parser.add_argument("--lots", type=int, default=200)
        parser.add_argument("--win-rate", type=float, default=0.52)
        parser.add_argument("--disposition", type=float, default=1.6)
        parser.add_argument("--open-fraction", type=float, default=0.20)
        parser.add_argument("--history-days", type=int, default=540)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--clear", action="store_true")

    def handle(self, *args, **opts):
        if not settings.DEBUG:
            raise CommandError(
                "seed_trades refuses to run with DEBUG=False. "
                "This command is for local dev and tests only."
            )

        rng = random.Random(opts["seed"])
        cfg = SeedConfig(
            n_lots=opts["lots"],
            win_probability=opts["win_rate"],
            disposition_ratio=opts["disposition"],
            open_fraction=opts["open_fraction"],
            history_days=opts["history_days"],
        )

        with transaction.atomic():
            user = self._get_user(opts["user"])
            self._ensure_securities()

            if opts["clear"]:
                TradeFill.objects.filter(user=user).delete()
                PositionLot.objects.filter(user=user).delete()

            lots = self._generate_lots(rng, cfg)
            self._persist(user, lots)

        self._report(user, cfg)

    # -- setup ---------------------------------------------------------

    def _get_user(self, label: str) -> AppUser:
        user, created = AppUser.objects.get_or_create(cognito_sub=label)
        if created:
            self.stdout.write(f"Created AppUser {label}")
        return user

    def _ensure_securities(self):
        for symbol, _, _ in UNIVERSE:
            Security.objects.get_or_create(
                symbol=symbol, defaults={"name": symbol, "is_active": True}
            )

    # -- generation ----------------------------------------------------

    def _generate_lots(self, rng: random.Random, cfg: SeedConfig) -> list[GeneratedLot]:
        """
        Build lots with known outcomes directly. Each lot's hold period and
        P&L are exactly what the winner/loser draw dictates -- nothing
        downstream reshapes them.
        """
        now = timezone.now()
        start = now - timedelta(days=cfg.history_days)
        avg_loser_hold = cfg.base_winner_hold_days * cfg.disposition_ratio
        lots: list[GeneratedLot] = []

        for _ in range(cfg.n_lots):
            symbol, base_price, _vol = rng.choice(UNIVERSE)
            buy_price = self._jitter(rng, base_price, 0.15)
            quantity = self._position_size(rng, buy_price)
            buy_offset = rng.uniform(0, cfg.history_days * 0.8)
            opened_at = start + timedelta(days=buy_offset)

            # Some positions stay open.
            if rng.random() < cfg.open_fraction:
                lots.append(GeneratedLot(
                    symbol=symbol, quantity=quantity, open_price=buy_price,
                    opened_at=opened_at, close_price=None, closed_at=None,
                    realized_pnl=None, hold_days=None,
                ))
                continue

            is_winner = rng.random() < cfg.win_probability

            # The disposition effect: losers held longer than winners.
            if is_winner:
                hold = max(1, round(rng.gauss(
                    cfg.base_winner_hold_days, cfg.base_winner_hold_days * 0.4)))
                ret = abs(rng.gauss(0.08, 0.05))
            else:
                hold = max(1, round(rng.gauss(
                    avg_loser_hold, avg_loser_hold * 0.4)))
                ret = -abs(rng.gauss(0.07, 0.05))

            closed_at = opened_at + timedelta(days=hold)
            if closed_at >= now:
                # Would close in the future -- leave open instead.
                lots.append(GeneratedLot(
                    symbol=symbol, quantity=quantity, open_price=buy_price,
                    opened_at=opened_at, close_price=None, closed_at=None,
                    realized_pnl=None, hold_days=None,
                ))
                continue

            close_price = (buy_price * (Decimal("1") + Decimal(str(ret)))).quantize(
                Decimal("0.0001"))
            realized_pnl = ((close_price - buy_price) * quantity).quantize(
                Decimal("0.0001"))

            lots.append(GeneratedLot(
                symbol=symbol, quantity=quantity, open_price=buy_price,
                opened_at=opened_at, close_price=close_price, closed_at=closed_at,
                realized_pnl=realized_pnl,
                # Recompute hold_days from the timestamps so it matches what
                # the real matcher would derive -- not the drawn value, which
                # could differ by rounding.
                hold_days=(closed_at - opened_at).days,
            ))

        return lots

    def _jitter(self, rng, base: Decimal, pct: float) -> Decimal:
        return (base * Decimal(str(1 + rng.uniform(-pct, pct)))).quantize(
            Decimal("0.0001"))

    def _position_size(self, rng, price: Decimal) -> Decimal:
        target = rng.uniform(500, 5000)
        return Decimal(max(1, int(target / float(price))))

    # -- persistence ---------------------------------------------------

    def _persist(self, user: AppUser, lots: list[GeneratedLot]):
        """
        Write PositionLot rows directly from the known pairs, plus a
        TradeFill for each buy and sell so the fill log looks realistic.
        The fills are NOT re-matched -- the lots are the source of truth
        here, by design (see module docstring).
        """
        counter = 0
        lot_objs = []
        fill_objs = []

        for gl in lots:
            sec_id = self._security_id(gl.symbol)

            lot_objs.append(PositionLot(
                user=user,
                security_id=sec_id,
                original_qty=gl.quantity,
                remaining_qty=Decimal("0") if gl.is_closed else gl.quantity,
                open_price=gl.open_price,
                opened_at=gl.opened_at,
                closed_at=gl.closed_at,
                close_price=gl.close_price,
                realized_pnl=gl.realized_pnl,
                hold_days=gl.hold_days,
            ))

            counter += 1
            fill_objs.append(TradeFill(
                user=user, security_id=sec_id, side="buy",
                quantity=gl.quantity, price=gl.open_price,
                filled_at=gl.opened_at, source_event_id=f"seed-{counter}",
            ))

            if gl.is_closed:
                counter += 1
                fill_objs.append(TradeFill(
                    user=user, security_id=sec_id, side="sell",
                    quantity=gl.quantity, price=gl.close_price,
                    filled_at=gl.closed_at, source_event_id=f"seed-{counter}",
                ))

        PositionLot.objects.bulk_create(lot_objs)
        TradeFill.objects.bulk_create(fill_objs)


    def _security_id(self, symbol: str) -> int:
        if symbol not in self._sec_cache:
            self._sec_cache[symbol] = Security.objects.get(symbol=symbol).id
        return self._sec_cache[symbol]

    # -- reporting -----------------------------------------------------

    def _report(self, user: AppUser, cfg: SeedConfig):
        lots = PositionLot.objects.filter(user=user)
        closed = list(lots.filter(closed_at__isnull=False))
        n_closed = len(closed)

        if n_closed == 0:
            self.stdout.write(self.style.WARNING("No closed lots generated."))
            return

        winners = [l for l in closed if l.realized_pnl and l.realized_pnl > 0]
        losers = [l for l in closed if l.realized_pnl and l.realized_pnl <= 0]

        win_hold = _avg(l.hold_days for l in winners)
        loss_hold = _avg(l.hold_days for l in losers)
        ratio = (loss_hold / win_hold) if win_hold else float("nan")

        self.stdout.write(self.style.SUCCESS("\nSeed complete."))
        self.stdout.write(f"  Open lots:         {lots.filter(closed_at__isnull=True).count()}")
        self.stdout.write(f"  Closed lots:       {n_closed}")
        self.stdout.write(f"  Win rate:          {len(winners)/n_closed:.1%} (target {cfg.win_probability:.1%})")
        self.stdout.write(f"  Avg winner hold:   {win_hold:.1f} days")
        self.stdout.write(f"  Avg loser hold:    {loss_hold:.1f} days")
        self.stdout.write(f"  Disposition ratio: {ratio:.2f} (target {cfg.disposition_ratio:.2f})")


def _avg(values) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0
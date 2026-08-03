"""
Sync trades from DynamoDB into Postgres, then rebuild lots.

Two stages:

  1. Ingest -- copy TRADE items from DynamoDB into TradeFill rows,
     idempotently. The source_event_id unique constraint (carrying the
     DynamoDB tradeId) means re-running never double-counts; new fills are
     inserted, seen fills are skipped.

  2. Match -- for each user whose fills changed, rebuild PositionLot rows
     from the complete fill history via the FIFO matcher.

DynamoDB stays the source of truth; Postgres is a derived analytical copy.
Nothing is ever written back to DynamoDB.

Usage:
    python manage.py sync_trades
    python manage.py sync_trades --user <cognito_sub>
    python manage.py sync_trades --rematch-only
"""

from __future__ import annotations

from collections import defaultdict

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.market.models import Security
from apps.trading.dynamo_reader import TradeReader, TradeItem
from apps.trading.lot_matcher import Fill, Side, match_fills
from apps.trading.models import AppUser, PositionLot, TradeFill


class Command(BaseCommand):
    help = "Sync trades from DynamoDB into Postgres and rebuild lots."

    def add_arguments(self, parser):
        parser.add_argument("--user", help="Limit to a single cognito sub")
        parser.add_argument("--rematch-only", action="store_true",
                            help="Skip DynamoDB; just rebuild lots from existing fills")
        parser.add_argument("--table", help="Override the DynamoDB table name")

    def handle(self, *args, **opts):
        if opts["rematch_only"]:
            users = self._all_user_ids(opts.get("user"))
            self._rematch(users)
            return

        table = opts.get("table") or getattr(settings, "USER_DATA_TABLE", "")
        if not table:
            raise CommandError("USER_DATA_TABLE not set and --table not given")

        reader = TradeReader(table)
        changed_users = self._ingest(reader, opts.get("user"))
        self._rematch(changed_users)

    # -- stage 1: ingest ----------------------------------------------

    def _ingest(self, reader: TradeReader, only_user: str | None) -> set[int]:
        """
        Copy new trade items into TradeFill. Returns the set of AppUser ids
        that gained fills, so only those users get rematched.
        """
        existing = set(TradeFill.objects.values_list("source_event_id", flat=True))

        by_user: dict[str, list[TradeItem]] = defaultdict(list)
        seen = skipped = 0

        for item in reader.all_trades():
            if only_user and item.user_sub != only_user:
                continue
            seen += 1
            if item.trade_id in existing:
                continue  # already synced
            by_user[item.user_sub].append(item)

        changed: set[int] = set()
        new_fills = 0

        for sub, items in by_user.items():
            user = self._get_user(sub)
            fills = []
            for it in items:
                sec = self._security_id(it.symbol)
                if sec is None:
                    skipped += 1
                    continue
                fills.append(TradeFill(
                    user=user, security_id=sec,
                    side=it.side, quantity=it.quantity, price=it.price,
                    filled_at=it.executed_at, source_event_id=it.trade_id,
                ))
            if fills:
                # ignore_conflicts guards the race where two syncs overlap:
                # the unique constraint on source_event_id absorbs dupes.
                TradeFill.objects.bulk_create(fills, ignore_conflicts=True)
                new_fills += len(fills)
                changed.add(user.id)

        self.stdout.write(
            f"Ingest: {seen} trade items seen, {new_fills} new fills, "
            f"{skipped} skipped (unknown symbol)."
        )
        return changed

    # -- stage 2: rematch ---------------------------------------------

    def _rematch(self, user_ids: set[int]):
        """
        Rebuild PositionLot rows for the given users from their full fill
        history. A full rebuild (rather than incremental) keeps the lots
        deterministic and lets a logic change re-derive everything.
        """
        if not user_ids:
            self.stdout.write("Rematch: no users to process.")
            return

        for user_id in user_ids:
            with transaction.atomic():
                self._rematch_user(user_id)

        self.stdout.write(f"Rematch: rebuilt lots for {len(user_ids)} user(s).")

    def _rematch_user(self, user_id: int):
        rows = (
            TradeFill.objects
            .filter(user_id=user_id)
            .select_related("security")
            .order_by("filled_at", "source_event_id")
        )

        fills = [
            Fill(
                fill_id=r.source_event_id,
                symbol=r.security.symbol,
                side=Side.BUY if r.side == "buy" else Side.SELL,
                quantity=r.quantity,
                price=r.price,
                executed_at=r.filled_at,
            )
            for r in rows
        ]

        result = match_fills(fills)

        # Replace the user's lots wholesale. Safe inside the transaction --
        # readers see either the old set or the new, never a partial.
        PositionLot.objects.filter(user_id=user_id).delete()

        sec_by_symbol = {
            s.symbol: s.id
            for s in Security.objects.filter(
                symbol__in={l.symbol for l in result.lots}
            )
        }

        PositionLot.objects.bulk_create([
            PositionLot(
                user_id=user_id,
                security_id=sec_by_symbol[lot.symbol],
                original_qty=lot.original_qty,
                remaining_qty=lot.remaining_qty,
                open_price=lot.open_price,
                opened_at=lot.opened_at,
                closed_at=lot.closed_at,
                close_price=lot.close_price,
                realized_pnl=lot.realized_pnl if lot.is_closed else None,
                hold_days=lot.hold_days,
            )
            for lot in result.lots
        ])

        if result.unmatched_sells:
            self.stderr.write(self.style.WARNING(
                f"  user {user_id}: {len(result.unmatched_sells)} unmatched sells "
                f"(fill history may have gaps)"
            ))

    # -- helpers ------------------------------------------------------

    def _all_user_ids(self, only_user: str | None) -> set[int]:
        qs = AppUser.objects.all()
        if only_user:
            qs = qs.filter(cognito_sub=only_user)
        return set(qs.values_list("id", flat=True))

    _user_cache: dict[str, AppUser]

    def _get_user(self, sub: str) -> AppUser:
        if not hasattr(self, "_user_cache"):
            self._user_cache = {}
        if sub not in self._user_cache:
            self._user_cache[sub], _ = AppUser.objects.get_or_create(cognito_sub=sub)
        return self._user_cache[sub]

    _sec_cache: dict[str, int | None]

    def _security_id(self, symbol: str) -> int | None:
        """
        Resolve a symbol to a security id. Unknown symbols return None and
        the fill is skipped -- the analytics can't use a symbol with no
        price history anyway, and ingest_bars is what creates securities.
        """
        if not hasattr(self, "_sec_cache"):
            self._sec_cache = {}
        if symbol not in self._sec_cache:
            sec = Security.objects.filter(symbol=symbol).first()
            self._sec_cache[symbol] = sec.id if sec else None
        return self._sec_cache[symbol]
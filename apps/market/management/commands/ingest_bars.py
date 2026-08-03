"""
Ingest daily price history from Polygon into the daily_bars table.

Polygon is the sole source for analytical price history. The app's
DynamoDB daily table is left untouched -- its incremental fetches produce
an inconsistent adjustment basis over time, which is fine for charts but
wrong for return calculations. This command re-fetches the full window per
symbol so every bar shares one adjustment basis.

Symbol universe: by default, the union of symbols the user has actually
traded or currently holds -- not the whole market. That is all the
analytics needs and keeps runs small.

Usage:
    python manage.py ingest_bars                      # traded + held symbols
    python manage.py ingest_bars --symbols AAPL MSFT  # explicit list
    python manage.py ingest_bars --years 3 --delay 0.2
"""

from __future__ import annotations

import time
from datetime import date, timedelta

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.db.models import Q

from apps.market.models import DailyBar, Security
from apps.trading.models import PositionLot, TradeFill
from apps.market.polygon_client import PolygonClient, PolygonError


class Command(BaseCommand):
    help = "Fetch daily bars from Polygon into daily_bars."

    def add_arguments(self, parser):
        parser.add_argument("--symbols", nargs="*", help="Explicit symbols to ingest")
        parser.add_argument("--years", type=int, default=3,
                            help="History depth in years (default 3)")
        parser.add_argument("--delay", type=float, default=0.0,
                            help="Seconds to sleep between symbols (unpaid tiers)")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **opts):
        api_key = getattr(settings, "POLYGON_API_KEY", "") or ""
        client = PolygonClient(api_key)

        symbols = opts["symbols"] or self._discover_symbols()
        if not symbols:
            raise CommandError(
                "No symbols to ingest. Trade something first, or pass --symbols."
            )

        end = date.today()
        # A calendar buffer over the requested trading-day window: weekends
        # and holidays mean ~365 calendar days per ~252 trading days.
        start = end - timedelta(days=int(opts["years"] * 365))

        self.stdout.write(
            f"Ingesting {len(symbols)} symbol(s) from {start} to {end}"
        )

        totals = {"symbols": 0, "bars": 0, "errors": 0}

        for symbol in symbols:
            try:
                n = self._ingest_symbol(client, symbol, start, end, opts["dry_run"])
                totals["bars"] += n
                totals["symbols"] += 1
                self.stdout.write(f"  {symbol}: {n} bars")
            except PolygonError as e:
                totals["errors"] += 1
                # One bad symbol must not abort the whole run.
                self.stderr.write(self.style.WARNING(f"  {symbol}: {e}"))

            if opts["delay"] > 0:
                time.sleep(opts["delay"])

        self.stdout.write(self.style.SUCCESS(
            f"\nDone. {totals['symbols']} symbols, {totals['bars']} bars, "
            f"{totals['errors']} errors."
        ))

    # -- symbol discovery ---------------------------------------------

    def _discover_symbols(self) -> list[str]:
        """
        Symbols the user has traded or currently holds. These are the only
        ones the analytics can reference, so they are the only ones worth
        the API calls.
        """
        traded = set(
            TradeFill.objects.values_list("security__symbol", flat=True).distinct()
        )
        held = set(
            PositionLot.objects.filter(closed_at__isnull=True)
            .values_list("security__symbol", flat=True).distinct()
        )
        return sorted(traded | held)

    # -- per-symbol ingest --------------------------------------------

    def _ingest_symbol(self, client, symbol, start, end, dry_run) -> int:
        bars = client.daily_bars(symbol, start, end)
        if dry_run:
            return len(bars)
        if not bars:
            return 0

        security = self._get_or_create_security(symbol)

        # Upsert on the (security, bar_date) unique constraint. Re-running
        # re-adjusts the whole series consistently rather than duplicating.
        objs = [
            DailyBar(
                security=security, bar_date=b.bar_date,
                open=b.open, high=b.high, low=b.low, close=b.close,
                adj_close=b.adj_close, volume=b.volume,
            )
            for b in bars
        ]

        with transaction.atomic():
            DailyBar.objects.bulk_create(
                objs,
                update_conflicts=True,
                unique_fields=["security", "bar_date"],
                update_fields=["open", "high", "low", "close", "adj_close", "volume"],
            )
            # Keep the security's coverage bounds current -- cheap metadata
            # the analytics reads before deciding what to include.
            security.first_bar_date = bars[0].bar_date
            security.last_bar_date = bars[-1].bar_date
            security.save(update_fields=["first_bar_date", "last_bar_date"])

        return len(bars)

    def _get_or_create_security(self, symbol: str) -> Security:
        security, _ = Security.objects.get_or_create(
            symbol=symbol, defaults={"name": symbol, "is_active": True}
        )
        return security
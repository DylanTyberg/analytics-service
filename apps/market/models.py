from decimal import Decimal

from django.core.validators import MinValueValidator
from django.db import models


class Security(models.Model):
    """
    Reference data for a tradable symbol.

    This is the dimension table -- small (a few thousand rows), rarely
    changing, and referenced by nearly everything else. It exists so that
    daily_bars and trade_fills can key off a compact integer FK instead of
    repeating the symbol string on millions of rows.
    """

    class AssetType(models.TextChoices):
        STOCK = "stock", "Stock"
        ETF = "etf", "ETF"
        INDEX = "index", "Index"

    symbol = models.CharField(
        max_length=12,
        unique=True,
        db_index=True,
        help_text="Ticker as used by the data provider, e.g. AAPL",
    )
    name = models.CharField(max_length=255, blank=True)
    asset_type = models.CharField(
        max_length=8,
        choices=AssetType.choices,
        default=AssetType.STOCK,
    )
    sector = models.CharField(max_length=64, blank=True, db_index=True)
    exchange = models.CharField(max_length=32, blank=True)

    # Delisted symbols stay in the table -- historical bars and closed trade
    # lots still reference them. Filter on this rather than deleting rows.
    is_active = models.BooleanField(default=True, db_index=True)

    first_bar_date = models.DateField(null=True, blank=True)
    last_bar_date = models.DateField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "securities"
        verbose_name_plural = "securities"
        ordering = ["symbol"]

    def __str__(self) -> str:
        return self.symbol


class DailyBar(models.Model):
    """
    One trading day of OHLCV data for one security.

    This is the fact table and the only large one: ~4,000 symbols x ~250
    trading days x N years. Every design choice below is about keeping
    range scans over (security, date) fast.

    Prices are DecimalField, not FloatField. Float would be faster but
    introduces binary rounding error that compounds through return and
    P&L calculations. Decimal is the correct choice for money.
    """

    security = models.ForeignKey(
        Security,
        on_delete=models.CASCADE,
        related_name="bars",
    )
    bar_date = models.DateField()

    open = models.DecimalField(max_digits=14, decimal_places=4)
    high = models.DecimalField(max_digits=14, decimal_places=4)
    low = models.DecimalField(max_digits=14, decimal_places=4)
    close = models.DecimalField(max_digits=14, decimal_places=4)

    # Split/dividend adjusted close. Return calculations must use this --
    # a 2:1 split would otherwise look like a -50% day.
    adj_close = models.DecimalField(
        max_digits=14,
        decimal_places=4,
        validators=[MinValueValidator(Decimal("0"))],
    )

    volume = models.BigIntegerField()

    class Meta:
        db_table = "daily_bars"
        constraints = [
            # One bar per symbol per day. Also the idempotency guarantee for
            # ingestion: re-running a backfill can upsert, never duplicate.
            models.UniqueConstraint(
                fields=["security", "bar_date"],
                name="uniq_daily_bar_security_date",
            ),
            models.CheckConstraint(
                condition=models.Q(high__gte=models.F("low")),
                name="daily_bar_high_gte_low",
            ),
        ]
        indexes = [
            # The workhorse index. Descending date because analytical queries
            # almost always want the most recent N days ("last 3 years of
            # returns for these 40 symbols"), so this lets Postgres walk
            # backwards from the tail without a sort.
            models.Index(
                fields=["security", "-bar_date"],
                name="idx_bar_security_date_desc",
            ),
            # Supports cross-sectional queries: "every symbol's close on
            # this date", used when aligning a portfolio into a returns matrix.
            models.Index(fields=["bar_date"], name="idx_bar_date"),
        ]
        ordering = ["security", "-bar_date"]
        get_latest_by = "bar_date"

    def __str__(self) -> str:
        return f"{self.security.symbol} {self.bar_date} {self.close}"
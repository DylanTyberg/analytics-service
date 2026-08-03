from django.db import models

from apps.market.models import Security

class AppUser(models.Model):
    
    cognito_sub = models.CharField(max_length=64, unique=True, db_index=True)
    first_seen = models.DateTimeField(auto_now_add=True)
    last_synced_at = models.DateTimeField(null=True, blank=True)
    
    class Meta: 
        db_table = "app_users"
        
    def __str__(self) -> str:
        return self.cognito_sub

class PositionLot(models.Model):
    
    user = models.ForeignKey(AppUser, on_delete=models.CASCADE, related_name="lots")
    security = models.ForeignKey(Security, on_delete=models.PROTECT, related_name="lots")
    
    original_qty = models.DecimalField(max_digits=18, decimal_places=6)
    remaining_qty = models.DecimalField(max_digits=18, decimal_places=6)
 
    open_price = models.DecimalField(max_digits=14, decimal_places=4)
    opened_at = models.DateTimeField(db_index=True)
    
    closed_at = models.DateTimeField(null=True, blank=True, db_index=True)
    close_price = models.DecimalField(
        max_digits=14, decimal_places=4, null=True, blank=True
    )
    
    realized_pnl = models.DecimalField(
        max_digits=18, decimal_places=4, null=True, blank=True
    )
    
    hold_days = models.IntegerField(null=True, blank=True)
    
    class Meta:
        db_table = "position_lots"
        indexes = [
            # The core behavioural query: all closed lots for a user,
            # partitioned by winner/loser. Partial index keeps it small by
            # excluding the open lots, which that query never touches.
            models.Index(
                fields=["user", "closed_at"],
                name="idx_lot_user_closed",
                condition=models.Q(closed_at__isnull=False),
            ),
            # FIFO matching during ingest: "oldest open lot for this
            # user+security". Also partial -- only open lots are candidates.
            models.Index(
                fields=["user", "security", "opened_at"],
                name="idx_lot_open_fifo",
                condition=models.Q(closed_at__isnull=True),
            ),
        ]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(remaining_qty__gte=0),
                name="lot_remaining_qty_non_negative",
            ),
            models.CheckConstraint(
                condition=models.Q(remaining_qty__lte=models.F("original_qty")),
                name="lot_remaining_lte_original",
            ),
        ]
        ordering = ["user", "security", "opened_at"]
 
    def __str__(self) -> str:
        state = "open" if self.closed_at is None else "closed"
        return f"{self.security_id} lot {self.pk} ({state})"
 
    @property
    def is_open(self) -> bool:
        return self.closed_at is None
    
class TradeFill(models.Model):
    """
    Raw, immutable trade event mirrored from DynamoDB.
 
    Never updated after insert. PositionLot is the derived, interpreted view
    of these events; this table is the audit trail you can always rebuild
    lots from if the matching logic changes.
    """
 
    class Side(models.TextChoices):
        BUY = "buy", "Buy"
        SELL = "sell", "Sell"
 
    user = models.ForeignKey(AppUser, on_delete=models.CASCADE, related_name="fills")
    security = models.ForeignKey(Security, on_delete=models.PROTECT, related_name="fills")
 
    # Which lot this fill opened (buys) or consumed (sells). Null until the
    # matcher has processed it, which is also how the ETL finds unprocessed
    # fills on the next run.
    lot = models.ForeignKey(
        PositionLot,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="fills",
    )
 
    side = models.CharField(max_length=4, choices=Side.choices)
    quantity = models.DecimalField(max_digits=18, decimal_places=6)
    price = models.DecimalField(max_digits=14, decimal_places=4)
    filled_at = models.DateTimeField(db_index=True)
 
    # The DynamoDB event id. The unique constraint on it is what makes the
    # ETL idempotent -- re-running a sync window cannot double-count a fill.
    source_event_id = models.CharField(max_length=128, unique=True)
    synced_at = models.DateTimeField(auto_now_add=True)
 
    class Meta:
        db_table = "trade_fills"
        indexes = [
            # Chronological replay per user+security, which is exactly the
            # order the FIFO matcher consumes fills in.
            models.Index(
                fields=["user", "security", "filled_at"],
                name="idx_fill_user_sec_time",
            ),
            # Finds fills the matcher hasn't handled yet.
            models.Index(
                fields=["lot"],
                name="idx_fill_unmatched",
                condition=models.Q(lot__isnull=True),
            ),
        ]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(quantity__gt=0),
                name="fill_quantity_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(price__gte=0),
                name="fill_price_non_negative",
            ),
        ]
        ordering = ["user", "security", "filled_at"]
 
    def __str__(self) -> str:
        return f"{self.side} {self.quantity} {self.security_id} @ {self.price}"
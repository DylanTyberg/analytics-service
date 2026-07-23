from django.db import models
from decimal import Decimal
from django.core.validators import MinValueValidator

class Security(models.Model):
    
    class AssetType(models.TextChoices):
        STOCK = "stock", "Stock"
        ETF = "etf", "ETF"
        INDEX = "index", "Index"
    
    symbol = models.CharField(
        max_length=12,
        unique=True,
        db_index=True,
        help_text="Ticer as used by data provider",
    )
    name = models.CharField(max_length=255, blank=True)
    asset_type = models.CharField(
        max_length=8,
        choices=AssetType.choices,
        default=AssetType.STOCK,
    )
    sector = models.CharField(max_length=64, blank=True, db_index=True)
    exchange = models.CharField(max_length=32, blank=True)
    
    
    is_active = models.BooleanField(default=True, db_index=True)
    
    first_bar_date = models.DateField(null=True, blank=True)
    last_bar_date = models.DateField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    

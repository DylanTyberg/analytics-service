from django.db import models

from apps.trading.models import AppUser


class AnalysisRun(models.Model):
    """
    One execution of an analytics job, and the parent for its results.

    Results are stored as rows rather than computed on demand for three
    reasons: the pandas work is expensive enough that recomputing per page
    load is wasteful, long-running jobs need a status the API can poll
    instead of blocking a request, and keeping history lets you chart how a
    user's risk profile changed over time rather than only seeing "now".

    The (user, run_type, as_of, lookback_days) tuple is the cache key --
    if a run with those inputs already succeeded, serve it instead of
    recomputing.
    """

    class RunType(models.TextChoices):
        RISK = "risk", "Portfolio risk"
        BEHAVIOR = "behavior", "Trading behaviour"

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"

    user = models.ForeignKey(AppUser, on_delete=models.CASCADE, related_name="runs")
    run_type = models.CharField(max_length=16, choices=RunType.choices)

    # The as-of date the analysis was run against, and how far back it
    # looked. Both are inputs to the result, so both belong in the cache key.
    as_of = models.DateField()
    lookback_days = models.IntegerField(default=756)  # ~3 trading years

    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
    )
    error_message = models.TextField(blank=True)

    # Which securities actually made it into the calculation. Stored because
    # a position can be silently dropped for want of price history, and you
    # want the result to record what it was actually computed from.
    securities_included = models.JSONField(default=list, blank=True)
    securities_excluded = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    duration_ms = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = "analysis_runs"
        indexes = [
            # Cache lookup: most recent successful run matching these inputs.
            models.Index(
                fields=["user", "run_type", "-as_of"],
                name="idx_run_cache_lookup",
                condition=models.Q(status="succeeded"),
            ),
        ]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.run_type} for {self.user_id} @ {self.as_of} ({self.status})"


class RiskMetrics(models.Model):
    """
    Portfolio structure metrics for one run.

    Everything here describes observed risk -- how volatile the portfolio
    has been, how concentrated it is, how far it has fallen. Nothing here
    forecasts anything, which is deliberate: descriptive metrics are
    defensible, predictions are not.
    """

    run = models.OneToOneField(
        AnalysisRun, on_delete=models.CASCADE, related_name="risk_metrics"
    )

    # Annualised standard deviation of portfolio returns.
    portfolio_vol = models.DecimalField(max_digits=12, decimal_places=6)
    # Same, for an equal-weight or benchmark comparison series.
    benchmark_vol = models.DecimalField(
        max_digits=12, decimal_places=6, null=True, blank=True
    )
    # Sensitivity to the benchmark: cov(p, b) / var(b).
    beta = models.DecimalField(max_digits=10, decimal_places=6, null=True, blank=True)

    # Historical VaR: the 5th percentile daily return. Negative by
    # convention -- a loss threshold, not a magnitude.
    var_95 = models.DecimalField(max_digits=12, decimal_places=6)
    # Mean of returns worse than VaR. Captures tail severity, which VaR
    # alone hides.
    expected_shortfall_95 = models.DecimalField(max_digits=12, decimal_places=6)

    # Worst peak-to-trough decline over the window.
    max_drawdown = models.DecimalField(max_digits=12, decimal_places=6)
    max_drawdown_start = models.DateField(null=True, blank=True)
    max_drawdown_end = models.DateField(null=True, blank=True)

    # Herfindahl-Hirschman index of position weights: sum of squared
    # weights. 1.0 is a single holding; 1/n is perfectly equal-weighted.
    hhi = models.DecimalField(max_digits=10, decimal_places=6)
    # 1 / HHI -- "this portfolio behaves like N equal positions". More
    # intuitive to display than HHI itself.
    effective_holdings = models.DecimalField(max_digits=10, decimal_places=4)

    # Mean of the off-diagonal correlations. High means the diversification
    # is nominal: many positions, one underlying bet.
    avg_correlation = models.DecimalField(
        max_digits=8, decimal_places=6, null=True, blank=True
    )

    annualised_return = models.DecimalField(
        max_digits=12, decimal_places=6, null=True, blank=True
    )
    sharpe_ratio = models.DecimalField(
        max_digits=10, decimal_places=6, null=True, blank=True
    )

    # An N x N matrix is genuinely a document, not a relation -- splitting
    # it into pairwise rows would be normalisation for its own sake.
    # Shape: {"symbols": [...], "matrix": [[...], ...]}
    correlation_matrix = models.JSONField(default=dict)
    # Per-position share of total portfolio variance, which is where the
    # useful "one position is 60% of your risk" insight comes from.
    risk_contributions = models.JSONField(default=dict)

    class Meta:
        db_table = "risk_metrics"
        verbose_name_plural = "risk metrics"


class BehaviorMetrics(models.Model):
    """
    What the user's trading decisions look like in aggregate.

    Computed from closed position lots. This is the part of the feature
    that no other portfolio project has, because it needs trade history
    that only this platform generates.
    """

    run = models.OneToOneField(
        AnalysisRun, on_delete=models.CASCADE, related_name="behavior_metrics"
    )

    # --- Disposition effect: selling winners early, holding losers ---
    avg_hold_days_winners = models.DecimalField(
        max_digits=10, decimal_places=2, null=True, blank=True
    )
    avg_hold_days_losers = models.DecimalField(
        max_digits=10, decimal_places=2, null=True, blank=True
    )
    # losers / winners. Above 1.0 means losers are held longer, the
    # classic disposition pattern.
    disposition_ratio = models.DecimalField(
        max_digits=10, decimal_places=4, null=True, blank=True
    )

    # --- Outcomes ---
    total_closed_lots = models.IntegerField(default=0)
    winning_lots = models.IntegerField(default=0)
    win_rate = models.DecimalField(
        max_digits=6, decimal_places=4, null=True, blank=True
    )
    avg_win_pct = models.DecimalField(
        max_digits=10, decimal_places=6, null=True, blank=True
    )
    avg_loss_pct = models.DecimalField(
        max_digits=10, decimal_places=6, null=True, blank=True
    )
    # avg_win / |avg_loss|. A low win rate is fine if this is high.
    payoff_ratio = models.DecimalField(
        max_digits=10, decimal_places=4, null=True, blank=True
    )

    # --- Activity ---
    total_fills = models.IntegerField(default=0)
    trades_per_month = models.DecimalField(
        max_digits=10, decimal_places=2, null=True, blank=True
    )
    turnover_ratio = models.DecimalField(
        max_digits=10, decimal_places=4, null=True, blank=True
    )

    # --- The honest scorecard ---
    # Realised return vs. simply having bought and held the same positions.
    # Frequently negative, which is the useful finding.
    realized_return = models.DecimalField(
        max_digits=12, decimal_places=6, null=True, blank=True
    )
    buy_hold_return = models.DecimalField(
        max_digits=12, decimal_places=6, null=True, blank=True
    )
    vs_buy_hold = models.DecimalField(
        max_digits=12, decimal_places=6, null=True, blank=True
    )

    # Distribution data for charting: hold-period buckets, monthly counts,
    # P&L histogram. Presentation detail, not something to query on.
    distributions = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "behavior_metrics"
        verbose_name_plural = "behavior metrics"
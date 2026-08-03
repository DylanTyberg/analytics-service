"""
Orchestration for analysis runs.

This is the layer between the HTTP views and everything below them. A view
should do almost nothing: validate the request, call one function here,
serialize the result. This module does the actual work of an analysis --
fetch data through the repositories, run the pure engine, persist an
AnalysisRun and its metrics, and handle caching.

The engine stays pure and the repositories stay dumb; the knowledge of how
to assemble a complete analysis lives here and only here. That is what
keeps each layer testable in isolation.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from decimal import Decimal

from django.db import transaction
from django.utils import timezone

from apps.analytics.engine import behavior as bh
from apps.analytics.engine import returns as ret
from apps.analytics.engine import risk
from apps.analytics.models import (
    AnalysisRun,
    BehaviorMetrics,
    RiskMetrics,
)
from apps.market import repositories as market_repo
from apps.trading import repositories as trading_repo
from apps.trading.models import AppUser

# How recent a succeeded run must be to serve from cache rather than
# recompute. Prices update daily, so a run from today is still valid.
CACHE_TTL = timedelta(hours=12)


class AnalysisError(Exception):
    """Raised when an analysis cannot be completed for a stated reason."""


# ---------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------


def get_or_run_risk(
    user: AppUser,
    *,
    as_of: date | None = None,
    lookback_days: int = 756,
    force: bool = False,
) -> AnalysisRun:
    """
    Return a risk analysis for the user, from cache if a fresh one exists.

    The cache key is (user, run_type, as_of, lookback_days). A prior
    succeeded run matching it, within CACHE_TTL, is returned untouched --
    the pandas work is expensive enough that recomputing per page load is
    wasteful.
    """
    as_of = as_of or timezone.now().date()

    if not force:
        cached = _find_cached(user, AnalysisRun.RunType.RISK, as_of, lookback_days)
        if cached:
            return cached

    return _run_risk(user, as_of, lookback_days)


def get_or_run_behavior(
    user: AppUser,
    *,
    as_of: date | None = None,
    force: bool = False,
) -> AnalysisRun:
    """Return a behavioural analysis, from cache if fresh."""
    as_of = as_of or timezone.now().date()

    if not force:
        cached = _find_cached(user, AnalysisRun.RunType.BEHAVIOR, as_of, 0)
        if cached:
            return cached

    return _run_behavior(user, as_of)


# ---------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------


def _find_cached(
    user: AppUser, run_type: str, as_of: date, lookback_days: int
) -> AnalysisRun | None:
    cutoff = timezone.now() - CACHE_TTL
    return (
        AnalysisRun.objects.filter(
            user=user,
            run_type=run_type,
            as_of=as_of,
            lookback_days=lookback_days,
            status=AnalysisRun.Status.SUCCEEDED,
            completed_at__gte=cutoff,
        )
        .order_by("-completed_at")
        .first()
    )


# ---------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------


@transaction.atomic
def _run_risk(user: AppUser, as_of: date, lookback_days: int) -> AnalysisRun:
    """
    Compute portfolio risk metrics for the user's current holdings.

    The AnalysisRun row is created first, in RUNNING state, so a failure
    leaves a FAILED record with its error rather than nothing -- the UI
    can show "analysis failed" instead of hanging, and you can see why.
    """
    run = AnalysisRun.objects.create(
        user=user,
        run_type=AnalysisRun.RunType.RISK,
        as_of=as_of,
        lookback_days=lookback_days,
        status=AnalysisRun.Status.RUNNING,
    )
    started = time.monotonic()

    try:
        weights_raw = trading_repo.current_holdings(user.id)
        if not weights_raw:
            raise AnalysisError("No open positions to analyse")

        symbols = list(weights_raw.keys())
        start = as_of - timedelta(days=int(lookback_days * 1.5))  # calendar buffer

        raw_prices = market_repo.price_frame(symbols, start, as_of)
        if raw_prices.empty:
            raise AnalysisError("No price history available for holdings")

        # Clean, recording what was dropped and why.
        prices, excluded = ret.align_price_frame(raw_prices)
        included = list(prices.columns)

        if len(included) < 2:
            raise AnalysisError(
                "Need at least two positions with sufficient history for "
                "correlation-based risk metrics"
            )

        # Weights only over what survived cleaning, valued at current price.
        latest = market_repo.latest_prices(included)
        weights = _dollar_weights(weights_raw, latest, included)

        returns = ret.simple_returns(prices)

        # Benchmark: equal-weight of the same holdings, a self-consistent
        # reference that needs no external index data.
        benchmark = returns.mean(axis=1)
        portfolio = ret.portfolio_returns(returns, weights)

        drawdown = risk.max_drawdown(portfolio)
        contrib = risk.risk_contributions(returns, weights)
        conc = risk.concentration(weights)

        RiskMetrics.objects.create(
            run=run,
            portfolio_vol=_dec(risk.annualised_portfolio_vol(returns, weights)),
            benchmark_vol=_dec(ret.annualise_volatility(benchmark)),
            beta=_dec(risk.beta(portfolio, benchmark)),
            var_95=_dec(risk.historical_var(portfolio)),
            expected_shortfall_95=_dec(risk.expected_shortfall(portfolio)),
            max_drawdown=_dec(drawdown.max_drawdown),
            max_drawdown_start=drawdown.start,
            max_drawdown_end=drawdown.end,
            hhi=_dec(conc.hhi),
            effective_holdings=_dec(conc.effective_holdings),
            avg_correlation=_dec(risk.average_correlation(returns)),
            annualised_return=_dec(ret.annualise_return(portfolio)),
            sharpe_ratio=_dec(risk.sharpe_ratio(portfolio)),
            correlation_matrix=_corr_json(risk.correlation_matrix(returns)),
            risk_contributions=contrib.contribution_pct,
        )

        _mark_succeeded(run, started, included, excluded)
        return run

    except AnalysisError as e:
        _mark_failed(run, started, str(e))
        return run
    except Exception as e:  # noqa: BLE001 - record anything unexpected too
        _mark_failed(run, started, f"unexpected error: {e}")
        raise


# ---------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------


@transaction.atomic
def _run_behavior(user: AppUser, as_of: date) -> AnalysisRun:
    run = AnalysisRun.objects.create(
        user=user,
        run_type=AnalysisRun.RunType.BEHAVIOR,
        as_of=as_of,
        lookback_days=0,
        status=AnalysisRun.Status.RUNNING,
    )
    started = time.monotonic()

    try:
        lots = trading_repo.closed_lots(user.id)
        if not lots:
            raise AnalysisError("No closed trades to analyse yet")

        opens = trading_repo.open_lots(user.id)

        disposition = bh.disposition_effect(lots)
        outcome = bh.outcomes(lots)

        # Current prices for the buy-hold comparison and turnover base.
        symbols = list({l.symbol for l in lots} | {o.symbol for o in opens})
        current = {
            s: Decimal(str(p)) for s, p in market_repo.latest_prices(symbols).items()
        }
        portfolio_value = _portfolio_value(opens, current)

        act = bh.activity(lots, opens, portfolio_value=portfolio_value)
        perf = bh.performance_vs_buy_hold(lots, current)

        BehaviorMetrics.objects.create(
            run=run,
            avg_hold_days_winners=_dec(disposition.avg_hold_winners),
            avg_hold_days_losers=_dec(disposition.avg_hold_losers),
            disposition_ratio=_dec(disposition.ratio),
            total_closed_lots=outcome.total_closed,
            winning_lots=outcome.winners,
            win_rate=_dec(outcome.win_rate),
            avg_win_pct=_dec(outcome.avg_win_pct),
            avg_loss_pct=_dec(outcome.avg_loss_pct),
            payoff_ratio=_dec(outcome.payoff_ratio),
            total_fills=outcome.total_closed * 2,  # rough: one buy, one sell each
            trades_per_month=_dec(act.trades_per_month),
            turnover_ratio=_dec(act.turnover_ratio),
            realized_return=_dec(perf.realized_return),
            buy_hold_return=_dec(perf.buy_hold_return),
            vs_buy_hold=_dec(perf.vs_buy_hold),
            distributions={
                "hold_period_buckets": bh.hold_period_buckets(lots),
                "monthly_trade_counts": bh.monthly_trade_counts(lots),
            },
        )

        _mark_succeeded(run, started, symbols, {})
        return run

    except AnalysisError as e:
        _mark_failed(run, started, str(e))
        return run
    except Exception as e:  # noqa: BLE001
        _mark_failed(run, started, f"unexpected error: {e}")
        raise


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _dollar_weights(
    quantities: dict[str, Decimal],
    prices: dict[str, float],
    included: list[str],
) -> dict[str, float]:
    """Market value per position, over the included symbols only."""
    values = {
        s: float(quantities[s]) * prices[s]
        for s in included
        if s in prices and s in quantities
    }
    total = sum(values.values())
    if total <= 0:
        return {s: 0.0 for s in included}
    return {s: v / total for s, v in values.items()}


def _portfolio_value(open_lots, current_prices: dict[str, Decimal]) -> float | None:
    total = 0.0
    seen = False
    for lot in open_lots:
        price = current_prices.get(lot.symbol)
        if price is None:
            continue
        total += float(price) * float(lot.quantity)
        seen = True
    return total if seen else None


def _corr_json(corr) -> dict:
    """Correlation DataFrame -> {"symbols": [...], "matrix": [[...]]}."""
    return {
        "symbols": list(corr.columns),
        "matrix": [[_round(v) for v in row] for row in corr.values.tolist()],
    }


def _dec(value) -> Decimal | None:
    """Float (or None/NaN) -> Decimal for storage. NaN becomes None."""
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
        return Decimal(str(round(float(value), 6)))
    except (ValueError, TypeError):
        return None


def _round(value):
    if value is None or value != value:
        return None
    return round(float(value), 6)


def _mark_succeeded(run, started, included, excluded):
    run.status = AnalysisRun.Status.SUCCEEDED
    run.completed_at = timezone.now()
    run.duration_ms = int((time.monotonic() - started) * 1000)
    run.securities_included = included
    run.securities_excluded = excluded
    run.save(update_fields=[
        "status", "completed_at", "duration_ms",
        "securities_included", "securities_excluded",
    ])


def _mark_failed(run, started, message):
    run.status = AnalysisRun.Status.FAILED
    run.completed_at = timezone.now()
    run.duration_ms = int((time.monotonic() - started) * 1000)
    run.error_message = message
    run.save(update_fields=[
        "status", "completed_at", "duration_ms", "error_message",
    ])
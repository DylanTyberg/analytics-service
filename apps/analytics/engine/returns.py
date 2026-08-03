"""
Return series construction.

Pure functions. No Django imports, no database access, no I/O. Everything
here takes a DataFrame or Series and returns one. That constraint is what
makes this module testable in milliseconds with hand-built inputs, and it
keeps the quantitative logic portable.

Convention throughout: a "price frame" is indexed by date with one column
per symbol, holding ADJUSTED closes. Using unadjusted prices here would
make a 2:1 split look like a -50% day and poison everything downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Trading days per year. Used to annualise daily statistics.
TRADING_DAYS = 252


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Period-over-period percentage change.

    Simple (arithmetic) returns aggregate correctly ACROSS assets at a
    point in time: a portfolio's return is the weighted sum of its
    holdings' simple returns. That property is why portfolio construction
    uses these rather than log returns.

    The first row is always NaN (no prior price) and is dropped.
    """
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Continuously compounded returns: ln(P_t / P_{t-1}).

    Log returns aggregate correctly THROUGH TIME -- a multi-day return is
    just the sum of daily log returns. Preferred for volatility work and
    anything involving compounding.

    The two return types are close for small daily moves and diverge on
    large ones, so don't mix them within a single calculation.
    """
    return np.log(prices / prices.shift(1)).iloc[1:]


def align_price_frame(
    prices: pd.DataFrame,
    min_observations: int = 60,
    max_missing_pct: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Clean a raw price frame into something safe to compute on.

    This is the unglamorous function that determines whether every
    downstream number is trustworthy. Real holdings include recent IPOs,
    thinly traded names, and symbols whose history starts mid-window --
    naively computing a correlation matrix across them produces numbers
    that look fine and mean nothing.

    Returns the cleaned frame plus a dict of {symbol: reason} for every
    column dropped, so the caller can record what was actually used
    rather than silently analysing a subset.
    """
    excluded: dict[str, str] = {}

    if prices.empty:
        return prices, excluded

    prices = prices.sort_index()

    for symbol in prices.columns:
        col = prices[symbol]
        observed = col.notna().sum()

        if observed < min_observations:
            excluded[symbol] = (
                f"insufficient history: {observed} of {min_observations} required"
            )
            continue

        # Gaps WITHIN the symbol's own lifespan, ignoring leading NaNs from
        # a later listing date.
        first_valid = col.first_valid_index()
        within = col.loc[first_valid:]
        missing_pct = within.isna().sum() / len(within) if len(within) else 1.0

        if missing_pct > max_missing_pct:
            excluded[symbol] = f"too many gaps: {missing_pct:.1%} missing"

    cleaned = prices.drop(columns=list(excluded.keys()))

    if cleaned.empty:
        return cleaned, excluded

    # Forward-fill short gaps (holidays, halts) but never backward-fill:
    # that would leak a future price into an earlier date.
    cleaned = cleaned.ffill()

    # Drop any leading rows where some symbol still has no price. This is
    # what truncates the window to the shortest common history -- the
    # correlation matrix needs every pair measured over the same dates.
    cleaned = cleaned.dropna()

    return cleaned, excluded


def portfolio_returns(
    returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
) -> pd.Series:
    """
    Collapse per-symbol returns into a single portfolio return series.

    Assumes fixed weights (a static snapshot of the current portfolio
    applied backwards), which is the standard approach for "what is the
    risk profile of what I hold right now". It is NOT a performance
    history -- it deliberately ignores when positions were actually
    opened. Use realised P&L from position lots for actual performance.

    Weights are renormalised to sum to 1 over whatever symbols survived
    cleaning, so an excluded position doesn't silently shrink the
    portfolio to less than fully invested.
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights, dtype=float)

    common = returns.columns.intersection(weights.index)
    if len(common) == 0:
        return pd.Series(dtype=float)

    w = weights.loc[common]
    total = w.sum()
    if total == 0:
        return pd.Series(dtype=float)
    w = w / total

    return (returns[common] * w).sum(axis=1)


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Growth of 1 unit invested, as a running product of (1 + r).

    The input series of a drawdown calculation and the basis for any
    equity curve chart.
    """
    return (1.0 + returns).cumprod()


def annualise_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """
    Geometric mean return, scaled to a year.

    Geometric rather than arithmetic: compounding means an asset that
    gains 50% then loses 50% has a positive arithmetic mean and a negative
    actual outcome. This reports the outcome.
    """
    returns = returns.dropna()
    if returns.empty:
        return float("nan")

    total_growth = float((1.0 + returns).prod())
    if total_growth <= 0:
        return -1.0

    years = len(returns) / periods_per_year
    if years <= 0:
        return float("nan")

    return total_growth ** (1.0 / years) - 1.0


def annualise_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS
) -> float:
    """
    Annualised standard deviation of returns.

    Scales by sqrt(periods) -- variance is additive over independent
    periods, so standard deviation grows with the square root of time.

    Uses the sample standard deviation (ddof=1), the correct estimator
    when working from a sample rather than a full population.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")

    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def rolling_volatility(
    returns: pd.Series, window: int = 21, periods_per_year: int = TRADING_DAYS
) -> pd.Series:
    """
    Trailing annualised volatility. Default window ~= one trading month.

    Useful as a chart series and as a feature for regime classification.
    """
    return returns.rolling(window=window, min_periods=window).std(
        ddof=1
    ) * np.sqrt(periods_per_year)
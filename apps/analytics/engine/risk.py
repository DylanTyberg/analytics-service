"""
Portfolio risk metrics.

Pure functions, same contract as returns.py: DataFrames and Series in,
numbers out. No Django, no I/O, no database.

Everything here is DESCRIPTIVE -- it measures how a portfolio has behaved
and how it is currently structured. Nothing forecasts anything. That is a
deliberate choice: descriptive risk metrics are defensible and verifiable,
return predictions are neither.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .returns import TRADING_DAYS, annualise_volatility


@dataclass
class DrawdownResult:
    max_drawdown: float
    start: date | None
    end: date | None


@dataclass
class ConcentrationResult:
    hhi: float
    effective_holdings: float
    largest_weight: float
    top_3_weight: float


@dataclass
class RiskContribution:
    """Per-position share of total portfolio variance."""

    weights: dict[str, float] = field(default_factory=dict)
    marginal: dict[str, float] = field(default_factory=dict)
    contribution_pct: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Tail risk
# ---------------------------------------------------------------------


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Value at Risk from the empirical distribution.

    The 5th percentile of observed daily returns at 95% confidence. Answers
    "on a bad day, how bad?" -- 5% of days were worse than this.

    Returned as a NEGATIVE number, by convention: it is a return threshold,
    not a loss magnitude.

    Historical rather than parametric because it makes no normality
    assumption. Financial returns have fat tails, so a normal-distribution
    VaR systematically understates risk. The cost is that it can only
    describe losses of a size already present in the window.
    """
    returns = returns.dropna()
    if returns.empty:
        return float("nan")

    return float(np.percentile(returns, (1.0 - confidence) * 100.0))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    VaR assuming normally distributed returns: mu + z * sigma.

    Included mainly as a comparison to the historical figure. When the two
    diverge sharply, the return distribution is non-normal -- which is
    itself the interesting finding, and worth surfacing.
    """
    from scipy.stats import norm  # local import: only needed here

    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")

    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    z = float(norm.ppf(1.0 - confidence))

    return mu + z * sigma


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Mean return of the days worse than VaR. Also called CVaR.

    VaR gives a threshold but says nothing about severity beyond it -- two
    portfolios can share a VaR while one has a far worse tail. Expected
    shortfall captures that, which is why regulators moved toward it.

    Also negative by convention, and always <= VaR.
    """
    returns = returns.dropna()
    if returns.empty:
        return float("nan")

    var = historical_var(returns, confidence)
    tail = returns[returns <= var]

    if tail.empty:
        return var

    return float(tail.mean())


# ---------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------


def max_drawdown(returns: pd.Series) -> DrawdownResult:
    """
    Worst peak-to-trough decline over the window.

    Arguably the risk metric investors actually feel: volatility is
    abstract, "you were down 34% from the high" is not.

    Computed from the cumulative growth curve -- track the running maximum,
    measure how far below it the curve falls, take the worst.
    """
    returns = returns.dropna()
    if returns.empty:
        return DrawdownResult(float("nan"), None, None)

    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    trough_idx = drawdown.idxmin()
    worst = float(drawdown.min())

    # The peak is the last date at or above the running max before the
    # trough -- i.e. where the decline started.
    before_trough = cumulative.loc[:trough_idx]
    peak_idx = before_trough.idxmax()

    def _as_date(x):
        if isinstance(x, pd.Timestamp):
            return x.date()
        return x if isinstance(x, date) else None

    return DrawdownResult(
        max_drawdown=worst,
        start=_as_date(peak_idx),
        end=_as_date(trough_idx),
    )


# ---------------------------------------------------------------------
# Benchmark-relative
# ---------------------------------------------------------------------


def beta(portfolio: pd.Series, benchmark: pd.Series) -> float:
    """
    Sensitivity to the benchmark: cov(p, b) / var(b).

    Beta of 1.2 means the portfolio has historically moved ~20% more than
    the benchmark in both directions. Measures systematic risk -- the part
    that cannot be diversified away.
    """
    aligned = pd.concat([portfolio, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")

    p, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    benchmark_var = float(b.var(ddof=1))

    if benchmark_var == 0:
        return float("nan")

    return float(p.cov(b) / benchmark_var)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Excess return per unit of volatility.

    Note the well-known flaw: standard deviation penalises upside and
    downside equally, so a portfolio that occasionally jumps sharply
    upward is scored as "risky". Reported because it is the common
    language, not because it is the best measure.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")

    periodic_rf = risk_free_rate / periods_per_year
    excess = returns - periodic_rf

    vol = float(excess.std(ddof=1))
    if vol < 1e-12:
        return float("nan")

    return float(excess.mean() / vol * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------
# Concentration and correlation
# ---------------------------------------------------------------------


def concentration(weights: dict[str, float] | pd.Series) -> ConcentrationResult:
    """
    How concentrated the portfolio is, by position size.

    HHI is the sum of squared weights: 1.0 for a single holding, 1/n for n
    equal positions. Its reciprocal -- the effective number of holdings --
    is far more legible to a user: "your 12 positions behave like 3.4
    equal ones".

    Note this measures SIZE concentration only. Ten equally weighted
    positions that are all regional banks score as well diversified here;
    the correlation metrics below are what catch that.
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights, dtype=float)

    weights = weights[weights > 0]
    if weights.empty:
        return ConcentrationResult(float("nan"), float("nan"), float("nan"), float("nan"))

    w = weights / weights.sum()
    hhi = float((w**2).sum())
    ordered = w.sort_values(ascending=False)

    return ConcentrationResult(
        hhi=hhi,
        effective_holdings=float(1.0 / hhi) if hhi > 0 else float("nan"),
        largest_weight=float(ordered.iloc[0]),
        top_3_weight=float(ordered.head(3).sum()),
    )


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation of return series."""
    return returns.corr()


def average_correlation(returns: pd.DataFrame) -> float:
    """
    Mean of the off-diagonal correlations.

    The diagonal is all 1.0 and would bias the mean upward, so it is
    excluded. A high value is the diversification warning: many positions,
    one underlying bet.
    """
    if returns.shape[1] < 2:
        return float("nan")

    corr = returns.corr().values
    upper = corr[np.triu_indices_from(corr, k=1)]
    upper = upper[~np.isnan(upper)]

    if upper.size == 0:
        return float("nan")

    return float(upper.mean())


def risk_contributions(
    returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
) -> RiskContribution:
    """
    Decompose portfolio variance by position.

    The most actionable output in this module, because weight and risk
    share diverge sharply: a 15% position in something volatile and
    correlated can drive 40% of total portfolio risk. Position size alone
    never reveals that.

    Uses Euler decomposition -- marginal contributions weight-averaged to
    exactly the portfolio's total variance, so the percentages sum to 100.
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights, dtype=float)

    common = returns.columns.intersection(weights.index)
    if len(common) == 0:
        return RiskContribution()

    w = weights.loc[common]
    if w.sum() == 0:
        return RiskContribution()
    w = w / w.sum()

    cov = returns[common].cov()
    portfolio_var = float(w.values @ cov.values @ w.values)

    if portfolio_var <= 0:
        return RiskContribution(weights=w.to_dict())

    # Marginal contribution of each position to portfolio variance.
    marginal = cov.values @ w.values
    contribution = w.values * marginal
    pct = contribution / portfolio_var

    return RiskContribution(
        weights={k: float(v) for k, v in w.items()},
        marginal={k: float(v) for k, v in zip(common, marginal)},
        contribution_pct={k: float(v) for k, v in zip(common, pct)},
    )


def annualised_portfolio_vol(
    returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Portfolio volatility computed from the covariance matrix:
    sqrt(w' * COV * w), annualised.

    Mathematically equivalent to taking the standard deviation of the
    weighted return series, but going through the covariance matrix makes
    the diversification effect explicit -- the result is lower than the
    weighted average of individual volatilities precisely because the
    off-diagonal correlations are below 1.
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights, dtype=float)

    common = returns.columns.intersection(weights.index)
    if len(common) == 0:
        return float("nan")

    w = weights.loc[common]
    if w.sum() == 0:
        return float("nan")
    w = w / w.sum()

    cov = returns[common].cov()
    variance = float(w.values @ cov.values @ w.values)

    if variance < 0:
        return float("nan")

    return float(np.sqrt(variance) * np.sqrt(periods_per_year))
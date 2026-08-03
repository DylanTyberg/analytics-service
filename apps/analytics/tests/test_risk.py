"""
Tests for the analytics engine.

No database, no Django test client, no fixtures -- these import pure
functions and feed them hand-built arrays. That is the payoff for keeping
engine/ free of Django imports: the whole suite runs in milliseconds.

The important cases are the known-answer ones, where the expected value is
derived by hand rather than snapshotted from a previous run. A snapshot
test only tells you the code still does what it did; a known-answer test
tells you it does the right thing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apps.analytics.engine import returns as ret
from apps.analytics.engine import risk


@pytest.fixture
def dates():
    return pd.date_range("2024-01-01", periods=100, freq="B")


# ---------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------


def test_simple_returns_known_answer():
    prices = pd.Series([100.0, 110.0, 99.0])
    result = ret.simple_returns(prices)

    # 100 -> 110 is +10%; 110 -> 99 is -10%.
    assert result.iloc[0] == pytest.approx(0.10)
    assert result.iloc[1] == pytest.approx(-0.10)
    # First row is dropped, so n-1 observations.
    assert len(result) == 2


def test_log_returns_sum_through_time():
    """Log returns are additive over time -- that is the whole point."""
    prices = pd.Series([100.0, 110.0, 121.0])
    logs = ret.log_returns(prices)

    total = float(logs.sum())
    expected = np.log(121.0 / 100.0)
    assert total == pytest.approx(expected)


def test_annualise_return_uses_geometric_mean():
    """
    +50% then -50% leaves you at 0.75, i.e. a real loss. The arithmetic
    mean would say 0%. This must report the loss.
    """
    returns = pd.Series([0.5, -0.5])
    result = ret.annualise_return(returns, periods_per_year=2)

    # One "year" of two periods: 1.5 * 0.5 = 0.75 -> -25%.
    assert result == pytest.approx(-0.25)


def test_annualise_volatility_scales_by_sqrt_time():
    rng = np.random.default_rng(42)
    daily = pd.Series(rng.normal(0, 0.01, 1000))

    annual = ret.annualise_volatility(daily, periods_per_year=252)
    expected = float(daily.std(ddof=1)) * np.sqrt(252)

    assert annual == pytest.approx(expected)


def test_portfolio_returns_weighted_sum():
    returns = pd.DataFrame({"A": [0.10, 0.00], "B": [0.00, 0.20]})
    result = ret.portfolio_returns(returns, {"A": 0.5, "B": 0.5})

    assert result.iloc[0] == pytest.approx(0.05)
    assert result.iloc[1] == pytest.approx(0.10)


def test_portfolio_returns_renormalises_weights():
    """
    Weights that don't sum to 1 get renormalised, so a position dropped
    during cleaning doesn't silently leave the portfolio under-invested.
    """
    returns = pd.DataFrame({"A": [0.10], "B": [0.20]})
    result = ret.portfolio_returns(returns, {"A": 1.0, "B": 1.0})

    assert result.iloc[0] == pytest.approx(0.15)


def test_align_drops_symbol_with_insufficient_history(dates):
    prices = pd.DataFrame(
        {
            "GOOD": np.linspace(100, 150, 100),
            "IPO": [np.nan] * 90 + list(np.linspace(50, 55, 10)),
        },
        index=dates,
    )

    cleaned, excluded = ret.align_price_frame(prices, min_observations=60)

    assert "GOOD" in cleaned.columns
    assert "IPO" not in cleaned.columns
    assert "IPO" in excluded
    assert "insufficient history" in excluded["IPO"]


def test_align_never_backfills():
    # Leading NaN (drop, don't backfill) + one interior gap that stays
    # under the 10% threshold, so the column survives to prove ffill works.
    col = [np.nan] + [100.0] * 5 + [np.nan] + [102.0] * 13   # 20 rows, 1 interior gap
    prices = pd.DataFrame({"A": col})
    cleaned, excluded = ret.align_price_frame(prices, min_observations=2)

    assert "A" not in excluded              # 1/19 ≈ 5.3% gaps, under 10%
    assert len(cleaned) == len(col) - 1     # only the leading NaN row dropped
    assert cleaned["A"].iloc[0] == 100.0    # first real value, not backfilled
    assert cleaned["A"].iloc[5] == 100.0    # interior gap ffilled to 100, never 102


# ---------------------------------------------------------------------
# Tail risk
# ---------------------------------------------------------------------


def test_historical_var_is_a_percentile():
    returns = pd.Series(np.linspace(-0.10, 0.10, 101))
    var = risk.historical_var(returns, confidence=0.95)

    # 5th percentile of a symmetric linear spread from -10% to +10%.
    assert var == pytest.approx(np.percentile(returns, 5))
    assert var < 0


def test_expected_shortfall_worse_than_var():
    """ES is the mean of the tail beyond VaR, so it must be <= VaR."""
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0005, 0.012, 2000))

    var = risk.historical_var(returns)
    es = risk.expected_shortfall(returns)

    assert es <= var
    assert es < 0


def test_expected_shortfall_catches_fat_tail():
    """
    Two series with a similar VaR but very different tail severity. This
    is exactly the case VaR alone hides.
    """
    mild = pd.Series([-0.02] * 5 + [0.01] * 95)
    severe = pd.Series([-0.50] * 5 + [0.01] * 95)

    assert risk.expected_shortfall(severe) < risk.expected_shortfall(mild)


# ---------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------


def test_max_drawdown_known_answer():
    """
    Growth curve: 1.0 -> 1.2 -> 0.9 -> 1.1
    Peak 1.2, trough 0.9 -> drawdown = (0.9 - 1.2) / 1.2 = -25%.
    """
    returns = pd.Series([0.20, -0.25, 0.2222222])
    result = risk.max_drawdown(returns)

    assert result.max_drawdown == pytest.approx(-0.25, abs=1e-4)


def test_max_drawdown_monotonic_series_is_zero():
    returns = pd.Series([0.01] * 50)
    result = risk.max_drawdown(returns)

    assert result.max_drawdown == pytest.approx(0.0)


# ---------------------------------------------------------------------
# Benchmark-relative
# ---------------------------------------------------------------------


def test_beta_of_identical_series_is_one():
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(0, 0.01, 500))

    assert risk.beta(series, series) == pytest.approx(1.0)


def test_beta_of_doubled_series_is_two():
    rng = np.random.default_rng(2)
    benchmark = pd.Series(rng.normal(0, 0.01, 500))
    portfolio = benchmark * 2.0

    assert risk.beta(portfolio, benchmark) == pytest.approx(2.0)


def test_sharpe_zero_volatility_returns_nan():
    constant = pd.Series([0.001] * 100)
    assert np.isnan(risk.sharpe_ratio(constant))


# ---------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------


def test_hhi_single_position_is_one():
    result = risk.concentration({"AAPL": 1.0})

    assert result.hhi == pytest.approx(1.0)
    assert result.effective_holdings == pytest.approx(1.0)


def test_hhi_four_equal_positions():
    """Four equal weights -> HHI = 4 * 0.25^2 = 0.25 -> 4 effective."""
    weights = {s: 0.25 for s in ("A", "B", "C", "D")}
    result = risk.concentration(weights)

    assert result.hhi == pytest.approx(0.25)
    assert result.effective_holdings == pytest.approx(4.0)


def test_effective_holdings_below_count_when_uneven():
    """
    Ten positions where one dominates should behave like far fewer than
    ten -- this is the number that is actually useful to show a user.
    """
    weights = {"BIG": 0.82}
    weights.update({f"S{i}": 0.02 for i in range(9)})

    result = risk.concentration(weights)
    assert result.effective_holdings < 2.0
    assert result.largest_weight == pytest.approx(0.82)


# ---------------------------------------------------------------------
# Correlation and risk contribution
# ---------------------------------------------------------------------


def test_average_correlation_excludes_diagonal():
    """
    Two perfectly correlated series: the only off-diagonal pair is 1.0.
    If the diagonal leaked in the answer would still be 1.0, so use an
    anti-correlated pair to make the test meaningful.
    """
    base = pd.Series(np.linspace(-0.05, 0.05, 100))
    returns = pd.DataFrame({"A": base, "B": -base})

    assert risk.average_correlation(returns) == pytest.approx(-1.0)


def test_average_correlation_single_column_is_nan():
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01]})
    assert np.isnan(risk.average_correlation(returns))


def test_risk_contributions_sum_to_one():
    """Euler decomposition is exact -- the shares must total 100%."""
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(
        {
            "A": rng.normal(0, 0.01, 500),
            "B": rng.normal(0, 0.02, 500),
            "C": rng.normal(0, 0.005, 500),
        }
    )
    weights = {"A": 0.4, "B": 0.4, "C": 0.2}

    result = risk.risk_contributions(returns, weights)
    total = sum(result.contribution_pct.values())

    assert total == pytest.approx(1.0)


def test_risk_share_exceeds_weight_for_volatile_position():
    """
    The headline insight: a position's share of RISK can far exceed its
    share of CAPITAL. Here B is 20% of the portfolio by weight but far
    more than 20% of its variance.
    """
    rng = np.random.default_rng(13)
    returns = pd.DataFrame(
        {
            "CALM": rng.normal(0, 0.002, 1000),
            "WILD": rng.normal(0, 0.050, 1000),
        }
    )
    weights = {"CALM": 0.8, "WILD": 0.2}

    result = risk.risk_contributions(returns, weights)

    assert result.contribution_pct["WILD"] > 0.90
    assert result.contribution_pct["CALM"] < 0.10


def test_diversification_lowers_portfolio_vol():
    """
    Portfolio volatility should be BELOW the weighted average of the
    individual volatilities whenever correlations are under 1. That gap
    is the diversification benefit, and it is why the covariance matrix
    is used rather than averaging.
    """
    rng = np.random.default_rng(17)
    returns = pd.DataFrame(
        {
            "A": rng.normal(0, 0.01, 1000),
            "B": rng.normal(0, 0.01, 1000),
        }
    )
    weights = {"A": 0.5, "B": 0.5}

    portfolio_vol = risk.annualised_portfolio_vol(returns, weights)
    avg_individual = (
        ret.annualise_volatility(returns["A"]) + ret.annualise_volatility(returns["B"])
    ) / 2.0

    assert portfolio_vol < avg_individual


def test_perfectly_correlated_gives_no_diversification():
    """The boundary case: correlation 1.0 means no benefit at all."""
    rng = np.random.default_rng(19)
    base = rng.normal(0, 0.01, 1000)
    returns = pd.DataFrame({"A": base, "B": base})

    portfolio_vol = risk.annualised_portfolio_vol(returns, {"A": 0.5, "B": 0.5})
    single_vol = ret.annualise_volatility(returns["A"])

    assert portfolio_vol == pytest.approx(single_vol, rel=1e-6)


# ---------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        risk.historical_var,
        risk.expected_shortfall,
        ret.annualise_return,
        ret.annualise_volatility,
    ],
)
def test_empty_series_returns_nan_not_crash(fn):
    """
    Real portfolios contain positions with no usable history. These should
    degrade to NaN, not raise -- a single bad symbol must not take down
    the whole analysis.
    """
    assert np.isnan(fn(pd.Series(dtype=float)))
"""Long-only stratejiler icin vektorize performans hesaplama motoru."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Vektorize backtest serileri ve ozet performans olcumleri."""

    equity_curve: pd.Series
    strategy_returns: pd.Series
    turnover: pd.Series
    drawdown: pd.Series
    metrics: dict[str, float]


def run_long_only_backtest(
    close: pd.Series,
    position: pd.Series,
    *,
    transaction_cost: float = 0.0005,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Pozisyon serisinden maliyet-sonrasi getiri ve performans olcumleri hesaplar."""
    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    prices = _validate_prices(close)
    positions = _validate_positions(position, prices.index)
    asset_returns = prices.pct_change().fillna(0.0)
    turnover = positions.diff().abs().fillna(positions.abs()).rename("turnover")
    strategy_returns = (positions * asset_returns - turnover * transaction_cost).rename("strategy_return")
    equity_curve = (1.0 + strategy_returns).cumprod().rename("equity")
    drawdown = (equity_curve.div(equity_curve.cummax()) - 1.0).rename("drawdown")

    return BacktestResult(
        equity_curve=equity_curve,
        strategy_returns=strategy_returns,
        turnover=turnover,
        drawdown=drawdown,
        metrics=_calculate_metrics(strategy_returns, equity_curve, drawdown, periods_per_year),
    )


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Sifir risksiz getiri varsayimiyla yilliklastirilmis Sharpe orani."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    values = pd.to_numeric(returns, errors="raise").dropna()
    volatility = values.std(ddof=1)
    if values.size < 2 or np.isclose(volatility, 0.0):
        return 0.0
    return float(np.sqrt(periods_per_year) * values.mean() / volatility)


def maximum_drawdown(equity_curve: pd.Series) -> float:
    """Bir sermaye egrisinin en buyuk tepe-dip dususunu dondurur."""
    values = pd.to_numeric(equity_curve, errors="raise").dropna()
    if values.empty:
        raise ValueError("equity_curve must contain at least one value")
    return float((values.div(values.cummax()) - 1.0).min())


def _validate_prices(close: pd.Series) -> pd.Series:
    if not isinstance(close, pd.Series) or close.empty:
        raise ValueError("close must be a non-empty pandas Series")
    prices = pd.to_numeric(close, errors="raise").astype(float)
    if prices.isna().any() or (prices <= 0).any():
        raise ValueError("close must contain only positive, non-null prices")
    if not prices.index.is_unique:
        raise ValueError("close index must be unique")
    return prices.rename("close")


def _validate_positions(position: pd.Series, index: pd.Index) -> pd.Series:
    if not isinstance(position, pd.Series):
        raise TypeError("position must be a pandas Series")
    positions = pd.to_numeric(position.reindex(index), errors="raise").astype(float)
    if positions.isna().any():
        raise ValueError("position must provide a value for every close observation")
    if ((positions < 0.0) | (positions > 1.0)).any():
        raise ValueError("long-only position values must be between 0 and 1")
    return positions.rename("position")


def _calculate_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    drawdown: pd.Series,
    periods_per_year: int,
) -> dict[str, float]:
    periods = len(returns)
    total_return = float(equity_curve.iloc[-1] - 1.0)
    annualized_return = 0.0
    if periods > 0 and equity_curve.iloc[-1] > 0:
        annualized_return = float(equity_curve.iloc[-1] ** (periods_per_year / periods) - 1.0)
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if periods > 1 else 0.0
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year),
        "max_drawdown": float(drawdown.min()),
    }

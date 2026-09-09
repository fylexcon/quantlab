"""SMA-RSI stratejisi icin zaman sirali walk-forward dogrulama."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult, run_long_only_backtest
from strategies.trend_following import SmaRsiConfig, sma_rsi_signals


SelectionMetric = Literal["sharpe_ratio", "total_return"]


@dataclass(frozen=True)
class WalkForwardConfig:
    """Egitim ve out-of-sample pencerelerinin gozlem sayisiyla tanimi."""

    train_periods: int
    test_periods: int
    anchored: bool = False

    def __post_init__(self) -> None:
        for name, value in (("train_periods", self.train_periods), ("test_periods", self.test_periods)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class WalkForwardWindow:
    """Bir pencerenin secilen parametreleri ve egitim skoru."""

    train_start: object
    train_end: object
    test_start: object
    test_end: object
    selected_config: SmaRsiConfig
    training_score: float


@dataclass(frozen=True)
class WalkForwardResult:
    """Tum out-of-sample donem performansi ve pencere karar kaydi."""

    backtest: BacktestResult
    windows: tuple[WalkForwardWindow, ...]
    selections: pd.DataFrame


def run_sma_rsi_walk_forward(
    close: pd.Series,
    candidates: Sequence[SmaRsiConfig],
    window_config: WalkForwardConfig,
    *,
    selection_metric: SelectionMetric = "sharpe_ratio",
    transaction_cost: float = 0.0005,
    periods_per_year: int = 252,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    slippage_atr_multiplier: float = 0.0,
    slippage_atr_window: int = 14,
) -> WalkForwardResult:
    """Her egitim penceresinde parametre secer, sonraki pencerede sinar.

    Pencereler varsayilan olarak kayan yapidadir. ``anchored=True`` ile egitim
    araligi ilk gozlemden itibaren genisler. Parametre secimi asla test donemi
    verisini kullanmaz; test sinyalleri yalnizca o ana kadarki fiyat gecmisiyle
    hesaplanir.
    """
    prices = _validate_close(close)
    configs = tuple(candidates)
    if not configs or not all(isinstance(config, SmaRsiConfig) for config in configs):
        raise ValueError("candidates must contain at least one SmaRsiConfig")
    if selection_metric not in ("sharpe_ratio", "total_return"):
        raise ValueError("selection_metric must be 'sharpe_ratio' or 'total_return'")
    if len(prices) < window_config.train_periods + window_config.test_periods:
        raise ValueError("close does not contain one complete train/test window")

    windows: list[WalkForwardWindow] = []
    test_price_chunks: list[pd.Series] = []
    test_position_chunks: list[pd.Series] = []
    selection_rows: list[dict[str, object]] = []
    step = 0

    while True:
        train_start = 0 if window_config.anchored else step * window_config.test_periods
        train_stop = window_config.train_periods + step * window_config.test_periods
        test_start = train_stop
        test_stop = test_start + window_config.test_periods
        if test_stop > len(prices):
            break

        train_close = prices.iloc[train_start:train_stop]
        selected_config, score = _select_config(
            train_close,
            configs,
            selection_metric=selection_metric,
            transaction_cost=transaction_cost,
            periods_per_year=periods_per_year,
            high=_slice_optional(high, train_close.index),
            low=_slice_optional(low, train_close.index),
            slippage_atr_multiplier=slippage_atr_multiplier,
            slippage_atr_window=slippage_atr_window,
        )
        history_signals = sma_rsi_signals(prices.iloc[:test_stop], selected_config)
        test_close = prices.iloc[test_start:test_stop]
        test_position = history_signals.loc[test_close.index, "position"]
        window = WalkForwardWindow(
            train_start=train_close.index[0],
            train_end=train_close.index[-1],
            test_start=test_close.index[0],
            test_end=test_close.index[-1],
            selected_config=selected_config,
            training_score=score,
        )
        windows.append(window)
        test_price_chunks.append(test_close)
        test_position_chunks.append(test_position)
        selection_rows.append(_selection_row(window))
        step += 1

    oos_close = pd.concat(test_price_chunks)
    oos_position = pd.concat(test_position_chunks)
    backtest = run_long_only_backtest(
        oos_close,
        oos_position,
        transaction_cost=transaction_cost,
        periods_per_year=periods_per_year,
        high=_slice_optional(high, oos_close.index),
        low=_slice_optional(low, oos_close.index),
        slippage_atr_multiplier=slippage_atr_multiplier,
        slippage_atr_window=slippage_atr_window,
    )
    return WalkForwardResult(
        backtest=backtest,
        windows=tuple(windows),
        selections=pd.DataFrame(selection_rows),
    )


def _select_config(
    close: pd.Series,
    candidates: tuple[SmaRsiConfig, ...],
    *,
    selection_metric: SelectionMetric,
    transaction_cost: float,
    periods_per_year: int,
    high: pd.Series | None,
    low: pd.Series | None,
    slippage_atr_multiplier: float,
    slippage_atr_window: int,
) -> tuple[SmaRsiConfig, float]:
    best_config = candidates[0]
    best_score = -np.inf
    for config in candidates:
        signals = sma_rsi_signals(close, config)
        result = run_long_only_backtest(
            close,
            signals["position"],
            transaction_cost=transaction_cost,
            periods_per_year=periods_per_year,
            high=high,
            low=low,
            slippage_atr_multiplier=slippage_atr_multiplier,
            slippage_atr_window=slippage_atr_window,
        )
        score = float(result.metrics[selection_metric])
        if score > best_score:
            best_config, best_score = config, score
    return best_config, best_score


def _selection_row(window: WalkForwardWindow) -> dict[str, object]:
    config = window.selected_config
    return {
        "train_start": window.train_start,
        "train_end": window.train_end,
        "test_start": window.test_start,
        "test_end": window.test_end,
        "short_window": config.short_window,
        "long_window": config.long_window,
        "rsi_window": config.rsi_window,
        "oversold": config.oversold,
        "overbought": config.overbought,
        "training_score": window.training_score,
    }


def _validate_close(close: pd.Series) -> pd.Series:
    if not isinstance(close, pd.Series) or close.empty:
        raise ValueError("close must be a non-empty pandas Series")
    prices = pd.to_numeric(close, errors="raise").astype(float)
    if prices.isna().any() or (prices <= 0).any() or not prices.index.is_unique:
        raise ValueError("close must have a unique index and positive, non-null prices")
    return prices.rename("close")


def _slice_optional(values: pd.Series | None, index: pd.Index) -> pd.Series | None:
    if values is None:
        return None
    if not isinstance(values, pd.Series):
        raise TypeError("high and low must be pandas Series when provided")
    return values.reindex(index)

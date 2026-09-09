"""Vektorize teknik analiz gostergeleri."""

from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Basit hareketli ortalamayi hesaplar."""
    _validate_window(window)
    prices = _as_float_series(close, "close")
    return prices.rolling(window=window, min_periods=window).mean().rename(f"sma_{window}")


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder usulu Relative Strength Index (RSI) hesaplar."""
    _validate_window(window)
    prices = _as_float_series(close, "close")
    delta = prices.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = average_gain / average_loss
    values = 100 - (100 / (1 + relative_strength))
    values = values.mask(average_loss.eq(0) & average_gain.gt(0), 100.0)
    values = values.mask(average_gain.eq(0) & average_loss.gt(0), 0.0)
    values = values.mask(average_gain.eq(0) & average_loss.eq(0), 50.0)
    return values.rename(f"rsi_{window}")


def macd(
    close: pd.Series,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> pd.DataFrame:
    """MACD cizgisi, sinyal cizgisi ve histogramini dondurur."""
    _validate_window(fast_window)
    _validate_window(slow_window)
    _validate_window(signal_window)
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window")

    prices = _as_float_series(close, "close")
    fast_ema = prices.ewm(span=fast_window, adjust=False, min_periods=fast_window).mean()
    slow_ema = prices.ewm(span=slow_window, adjust=False, min_periods=slow_window).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(
        span=signal_window,
        adjust=False,
        min_periods=signal_window,
    ).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_line - signal_line,
        }
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder Average True Range (ATR) hesaplar."""
    _validate_window(window)
    high_prices = _as_float_series(high, "high")
    low_prices = _as_float_series(low, "low")
    close_prices = _as_float_series(close, "close")
    if not (high_prices.index.equals(low_prices.index) and high_prices.index.equals(close_prices.index)):
        raise ValueError("high, low, and close must share the same index")

    previous_close = close_prices.shift(1)
    true_range = pd.concat(
        [
            high_prices - low_prices,
            (high_prices - previous_close).abs(),
            (low_prices - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean().rename(f"atr_{window}")


def _as_float_series(values: pd.Series, name: str) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if values.empty:
        raise ValueError(f"{name} must not be empty")
    return pd.to_numeric(values, errors="raise").astype(float)


def _validate_window(window: int) -> None:
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("window must be a positive integer")

"""SMA ve RSI ile long-only trend takip sinyalleri."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from indicators.technical import rsi, sma


@dataclass(frozen=True)
class SmaRsiConfig:
    """SMA-RSI stratejisinin parametreleri."""

    short_window: int = 20
    long_window: int = 50
    rsi_window: int = 14
    oversold: float = 30.0
    overbought: float = 70.0

    def __post_init__(self) -> None:
        if self.short_window <= 0 or self.long_window <= 0 or self.rsi_window <= 0:
            raise ValueError("All indicator windows must be positive")
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be smaller than long_window")
        if not 0 <= self.oversold < self.overbought <= 100:
            raise ValueError("RSI thresholds must satisfy 0 <= oversold < overbought <= 100")


def sma_rsi_signals(close: pd.Series, config: SmaRsiConfig | None = None) -> pd.DataFrame:
    """SMA trendi ve RSI filtresiyle gecikmeli long-only pozisyonlar uretir.

    ``raw_signal`` bugunku kapanisla olusan karari, ``position`` ise bu kararin
    bir sonraki donemde uygulanmis halini tasir. Bu gecikme look-ahead bias'i
    onler.
    """
    settings = config or SmaRsiConfig()
    prices = pd.to_numeric(close, errors="raise").astype(float).rename("close")
    if prices.empty:
        raise ValueError("close must not be empty")

    short_sma = sma(prices, settings.short_window)
    long_sma = sma(prices, settings.long_window)
    rsi_values = rsi(prices, settings.rsi_window)
    raw_signal = (
        (short_sma > long_sma)
        & (rsi_values >= settings.oversold)
        & (rsi_values <= settings.overbought)
    ).astype(float).rename("raw_signal")
    position = raw_signal.shift(1).fillna(0.0).rename("position")

    return pd.DataFrame(
        {
            "close": prices,
            short_sma.name: short_sma,
            long_sma.name: long_sma,
            rsi_values.name: rsi_values,
            "raw_signal": raw_signal,
            "position": position,
        }
    )

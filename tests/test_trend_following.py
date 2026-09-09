import pandas as pd

from strategies.trend_following import SmaRsiConfig, sma_rsi_signals


def test_signal_is_applied_on_the_next_bar() -> None:
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    config = SmaRsiConfig(
        short_window=2,
        long_window=3,
        rsi_window=2,
        oversold=0.0,
        overbought=100.0,
    )

    signals = sma_rsi_signals(close, config)

    assert signals.loc[2, "raw_signal"] == 1.0
    assert signals.loc[2, "position"] == 0.0
    assert signals.loc[3, "position"] == 1.0

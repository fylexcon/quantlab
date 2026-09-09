import pandas as pd
import pytest

from indicators.technical import atr, macd, rsi, sma


def test_sma_requires_full_window() -> None:
    close = pd.Series([10.0, 11.0, 12.0, 13.0])

    result = sma(close, 3)

    assert result.isna().tolist() == [True, True, False, False]
    assert result.iloc[-1] == 12.0


def test_rsi_of_persistent_rise_is_100() -> None:
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])

    result = rsi(close, window=2)

    assert result.iloc[-1] == 100.0


def test_macd_produces_expected_columns() -> None:
    close = pd.Series(range(1, 40), dtype=float)

    result = macd(close)

    assert result.columns.tolist() == ["macd", "signal", "histogram"]
    assert result["macd"].notna().sum() > 0


def test_atr_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="same index"):
        atr(pd.Series([3.0, 4.0]), pd.Series([1.0, 2.0]), pd.Series([2.0, 3.0], index=[1, 2]))

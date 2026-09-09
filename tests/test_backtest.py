import pandas as pd
import pytest

from backtest.engine import maximum_drawdown, run_long_only_backtest


def test_long_only_backtest_calculates_equity_and_drawdown() -> None:
    close = pd.Series([100.0, 110.0, 99.0, 108.0])
    position = pd.Series([0.0, 1.0, 1.0, 0.0])

    result = run_long_only_backtest(close, position, transaction_cost=0.0)

    assert result.equity_curve.iloc[-1] == pytest.approx(0.99)
    assert result.metrics["total_return"] == pytest.approx(-0.01)
    assert result.metrics["max_drawdown"] == pytest.approx(-0.1)


def test_maximum_drawdown_rejects_empty_series() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        maximum_drawdown(pd.Series(dtype=float))


def test_positions_must_be_long_only() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_long_only_backtest(pd.Series([100.0, 101.0]), pd.Series([0.0, -1.0]))


def test_atr_slippage_is_charged_on_position_changes() -> None:
    close = pd.Series([100.0, 110.0, 99.0, 108.0])
    high = pd.Series([102.0, 112.0, 101.0, 110.0])
    low = pd.Series([98.0, 108.0, 97.0, 106.0])
    position = pd.Series([0.0, 1.0, 1.0, 0.0])

    result = run_long_only_backtest(
        close,
        position,
        transaction_cost=0.0,
        high=high,
        low=low,
        slippage_atr_multiplier=0.5,
        slippage_atr_window=2,
    )

    assert result.slippage_cost.iloc[0] == 0.0
    assert result.slippage_cost.iloc[1] > 0.0
    assert result.slippage_cost.iloc[2] == 0.0
    assert result.metrics["total_slippage_cost"] == pytest.approx(result.slippage_cost.sum())


def test_atr_slippage_requires_complete_ohlc_data() -> None:
    with pytest.raises(ValueError, match="high and low are required"):
        run_long_only_backtest(
            pd.Series([100.0, 101.0]),
            pd.Series([0.0, 1.0]),
            slippage_atr_multiplier=0.5,
        )

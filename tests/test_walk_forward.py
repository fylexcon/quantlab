import numpy as np
import pandas as pd
import pytest

from backtest.walk_forward import WalkForwardConfig, run_sma_rsi_walk_forward
from strategies.trend_following import SmaRsiConfig


def test_walk_forward_returns_contiguous_out_of_sample_backtest() -> None:
    close = pd.Series(
        np.linspace(100.0, 130.0, 24),
        index=pd.date_range("2024-01-01", periods=24, freq="B"),
    )
    candidates = (
        SmaRsiConfig(short_window=2, long_window=3, rsi_window=2, oversold=0.0, overbought=100.0),
        SmaRsiConfig(short_window=3, long_window=5, rsi_window=2, oversold=0.0, overbought=100.0),
    )

    result = run_sma_rsi_walk_forward(
        close,
        candidates,
        WalkForwardConfig(train_periods=8, test_periods=4),
        transaction_cost=0.0,
    )

    assert len(result.windows) == 4
    assert result.backtest.equity_curve.index.equals(close.index[8:])
    assert result.selections["test_start"].iloc[0] == close.index[8]
    assert set(result.selections["short_window"]) <= {2, 3}


def test_walk_forward_requires_complete_initial_window() -> None:
    close = pd.Series([100.0, 101.0, 102.0])
    candidate = SmaRsiConfig(short_window=1, long_window=2, rsi_window=1, oversold=0.0, overbought=100.0)

    with pytest.raises(ValueError, match="complete train/test window"):
        run_sma_rsi_walk_forward(close, [candidate], WalkForwardConfig(train_periods=2, test_periods=2))

"""QAOA optimizer ve katman benchmark'ini komut satirindan calistirir."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantum.benchmark import benchmark_qaoa_portfolio


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark QAOA portfolio optimizer settings.")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--risk-aversion", type=float, default=0.5)
    parser.add_argument("--reps", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--optimizers", nargs="+", default=["COBYLA", "SPSA"])
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    arguments = parser.parse_args()

    expected_returns = pd.Series({"AAPL": 0.12, "MSFT": 0.10, "NVDA": 0.16, "JNJ": 0.07})
    covariance = pd.DataFrame(
        [[0.040, 0.018, 0.025, 0.006], [0.018, 0.030, 0.020, 0.005], [0.025, 0.020, 0.070, 0.004], [0.006, 0.005, 0.004, 0.020]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )
    report = benchmark_qaoa_portfolio(
        expected_returns,
        covariance,
        budget=arguments.budget,
        risk_aversion=arguments.risk_aversion,
        reps_values=arguments.reps,
        optimizers=arguments.optimizers,
        maxiter=arguments.maxiter,
        seed=arguments.seed,
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()

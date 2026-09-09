"""QAOA katman ve klasik optimize edici karsilastirmasi."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import pandas as pd

from quantum.portfolio_qaoa import (
    SUPPORTED_OPTIMIZERS,
    PortfolioOptimizationResult,
    QAOAPortfolioConfig,
    solve_qaoa_portfolio,
)


def benchmark_qaoa_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    budget: int,
    risk_aversion: float = 0.5,
    reps_values: Sequence[int] = (1, 2),
    optimizers: Sequence[str] = ("COBYLA", "SPSA"),
    maxiter: int = 50,
    seed: int | None = 123,
) -> pd.DataFrame:
    """Her QAOA katman/optimize edici cifti icin cozum ve sure kaydi uretir.

    Her kosul ayni ``seed`` ile baslatilir. QAOA olasiliksal oldugu icin daha
    saglam bir karsilastirma icin farkli seed degerleriyle birden fazla calisma
    yapip sonuclari disarida gruplamak tavsiye edilir.
    """
    normalized_reps = tuple(reps_values)
    normalized_optimizers = tuple(optimizer.upper() for optimizer in optimizers)
    if not normalized_reps or any(not isinstance(reps, int) or isinstance(reps, bool) or reps <= 0 for reps in normalized_reps):
        raise ValueError("reps_values must contain positive integers")
    if not normalized_optimizers or any(optimizer not in SUPPORTED_OPTIMIZERS for optimizer in normalized_optimizers):
        supported = ", ".join(sorted(SUPPORTED_OPTIMIZERS))
        raise ValueError(f"optimizers must contain only: {supported}")

    rows: list[dict[str, object]] = []
    for optimizer in normalized_optimizers:
        for reps in normalized_reps:
            config = QAOAPortfolioConfig(
                budget=budget,
                risk_aversion=risk_aversion,
                reps=reps,
                maxiter=maxiter,
                optimizer=optimizer,
                seed=seed,
            )
            started_at = perf_counter()
            result = solve_qaoa_portfolio(expected_returns, covariance, config)
            rows.append(_result_row(config, result, perf_counter() - started_at))
    return pd.DataFrame(rows)


def _result_row(
    config: QAOAPortfolioConfig,
    result: PortfolioOptimizationResult,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "optimizer": config.optimizer,
        "reps": config.reps,
        "maxiter": config.maxiter,
        "selected_assets": ",".join(result.selected_assets),
        "objective_value": result.objective_value,
        "status": result.status,
        "elapsed_seconds": elapsed_seconds,
    }

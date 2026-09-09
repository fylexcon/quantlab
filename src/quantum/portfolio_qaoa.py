"""Sabit sayida varlik secimi icin QAOA portfoy optimizasyonu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class QiskitDependencyError(ImportError):
    """Kuantum calisma zamani paketleri kurulu olmadiginda yukseltilir."""


@dataclass(frozen=True)
class QAOAPortfolioConfig:
    """QAOA cozucusu ve portfoy kisitlari icin ayarlar."""

    budget: int = 2
    risk_aversion: float = 0.5
    reps: int = 2
    maxiter: int = 100
    seed: int | None = 123

    def __post_init__(self) -> None:
        if not isinstance(self.budget, int) or isinstance(self.budget, bool) or self.budget <= 0:
            raise ValueError("budget must be a positive integer")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")
        if not isinstance(self.reps, int) or self.reps <= 0:
            raise ValueError("reps must be a positive integer")
        if not isinstance(self.maxiter, int) or self.maxiter <= 0:
            raise ValueError("maxiter must be a positive integer")


@dataclass(frozen=True)
class PortfolioOptimizationResult:
    """Secilen varliklar ve ikili secimden turetilen esit agirliklar."""

    selected_assets: tuple[str, ...]
    weights: pd.Series
    objective_value: float
    status: str


def build_portfolio_problem(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    budget: int,
    risk_aversion: float = 0.5,
) -> Any:
    """QAOA icin ikili, sabit-butceli Markowitz problemi olusturur.

    Maksimize edilen amac ``mu.T @ x - risk_aversion * x.T @ Sigma @ x``
    seklindedir. ``x`` ikili secim vektorudur ve ``sum(x) == budget`` kisiti
    tasir; cozum agirliklari daha sonra secilen varliklara esit dagitilir.
    """
    _validate_inputs(expected_returns, covariance, budget, risk_aversion)
    QuadraticProgram = _quadratic_program_class()
    assets = [str(asset) for asset in expected_returns.index]
    problem = QuadraticProgram("qaoa_portfolio")
    for asset in assets:
        problem.binary_var(asset)

    linear = {asset: float(expected_returns.iloc[position]) for position, asset in enumerate(assets)}
    quadratic = {
        (row_asset, column_asset): -risk_aversion * float(covariance.iloc[row, column])
        for row, row_asset in enumerate(assets)
        for column, column_asset in enumerate(assets)
        if not np.isclose(covariance.iloc[row, column], 0.0)
    }
    problem.maximize(linear=linear, quadratic=quadratic)
    problem.linear_constraint(
        linear={asset: 1.0 for asset in assets},
        sense="==",
        rhs=budget,
        name="budget",
    )
    return problem


def solve_qaoa_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    config: QAOAPortfolioConfig | None = None,
) -> PortfolioOptimizationResult:
    """QAOA ile optimum varlik sepetini cozer ve esit agirliklara donusturur."""
    settings = config or QAOAPortfolioConfig()
    problem = build_portfolio_problem(
        expected_returns,
        covariance,
        budget=settings.budget,
        risk_aversion=settings.risk_aversion,
    )
    if settings.budget > len(expected_returns):
        raise ValueError("budget cannot exceed the number of assets")

    try:
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_algorithms.utils import algorithm_globals
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise QiskitDependencyError("Install qiskit, qiskit-algorithms, and qiskit-optimization") from error

    if settings.seed is not None:
        algorithm_globals.random_seed = settings.seed
    qaoa = QAOA(
        sampler=_create_sampler(),
        optimizer=COBYLA(maxiter=settings.maxiter),
        reps=settings.reps,
    )
    result = MinimumEigenOptimizer(qaoa).solve(problem)
    assets = [variable.name for variable in problem.variables]
    selection = pd.Series(np.rint(result.x).astype(int), index=assets, name="selected")
    selected_assets = tuple(selection[selection.eq(1)].index)
    if len(selected_assets) != settings.budget:
        raise RuntimeError("QAOA result did not satisfy the portfolio budget constraint")
    weights = selection.astype(float).div(settings.budget).rename("weight")

    return PortfolioOptimizationResult(
        selected_assets=selected_assets,
        weights=weights,
        objective_value=float(result.fval),
        status=str(result.status),
    )


def _quadratic_program_class() -> Any:
    try:
        from qiskit_optimization import QuadraticProgram
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise QiskitDependencyError("Install qiskit-optimization to build a QAOA portfolio problem") from error
    return QuadraticProgram


def _create_sampler() -> Any:
    """Qiskit 1.x ve sonraki ilkel API'leriyle uyumlu bir sampler uretir."""
    try:
        from qiskit.primitives import StatevectorSampler
    except ImportError:  # Qiskit 1.x V1 primitive API
        try:
            from qiskit.primitives import Sampler
        except ImportError as error:  # pragma: no cover - dependency boundary
            raise QiskitDependencyError("A Qiskit sampler primitive is required") from error
        return Sampler()
    return StatevectorSampler(default_shots=1024)


def _validate_inputs(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    budget: int,
    risk_aversion: float,
) -> None:
    if not isinstance(expected_returns, pd.Series) or expected_returns.empty:
        raise ValueError("expected_returns must be a non-empty pandas Series")
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a pandas DataFrame")
    if not expected_returns.index.is_unique or not covariance.index.is_unique:
        raise ValueError("asset labels must be unique")
    if len({str(asset) for asset in expected_returns.index}) != len(expected_returns):
        raise ValueError("asset labels must remain unique when converted to strings")
    if not expected_returns.index.equals(covariance.index) or not covariance.index.equals(covariance.columns):
        raise ValueError("covariance index and columns must match expected_returns index exactly")
    if not isinstance(budget, int) or isinstance(budget, bool) or not 0 < budget <= len(expected_returns):
        raise ValueError("budget must be between 1 and the number of assets")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")

    values = covariance.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(expected_returns.to_numpy(dtype=float)).all():
        raise ValueError("expected returns and covariance must contain finite values")
    if not np.allclose(values, values.T):
        raise ValueError("covariance must be symmetric")

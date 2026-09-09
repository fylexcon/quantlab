import pandas as pd

from quantum.benchmark import benchmark_qaoa_portfolio
from quantum.portfolio_qaoa import PortfolioOptimizationResult


def test_benchmark_returns_one_row_for_each_optimizer_and_depth(monkeypatch) -> None:
    def fake_solver(*_args, **_kwargs) -> PortfolioOptimizationResult:
        return PortfolioOptimizationResult(
            selected_assets=("AAA",),
            weights=pd.Series({"AAA": 1.0, "BBB": 0.0}),
            objective_value=0.42,
            status="SUCCESS",
        )

    monkeypatch.setattr("quantum.benchmark.solve_qaoa_portfolio", fake_solver)
    expected_returns = pd.Series({"AAA": 0.10, "BBB": 0.08})
    covariance = pd.DataFrame([[0.04, 0.01], [0.01, 0.03]], index=expected_returns.index, columns=expected_returns.index)

    report = benchmark_qaoa_portfolio(
        expected_returns,
        covariance,
        budget=1,
        reps_values=(1, 2),
        optimizers=("COBYLA", "SPSA"),
        maxiter=3,
    )

    assert len(report) == 4
    assert report["optimizer"].tolist() == ["COBYLA", "COBYLA", "SPSA", "SPSA"]
    assert report["reps"].tolist() == [1, 2, 1, 2]
    assert report["objective_value"].eq(0.42).all()

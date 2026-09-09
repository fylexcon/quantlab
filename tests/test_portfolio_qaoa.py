import pandas as pd
import pytest

from quantum.portfolio_qaoa import QAOAPortfolioConfig, build_portfolio_problem


def test_config_rejects_budget_larger_than_problem_later() -> None:
    config = QAOAPortfolioConfig(budget=3)

    assert config.budget == 3


def test_config_normalizes_supported_optimizer_name() -> None:
    config = QAOAPortfolioConfig(optimizer="spsa")

    assert config.optimizer == "SPSA"


def test_config_rejects_unknown_optimizer() -> None:
    with pytest.raises(ValueError, match="optimizer"):
        QAOAPortfolioConfig(optimizer="nelder-mead")


def test_problem_has_binary_assets_and_budget_constraint() -> None:
    pytest.importorskip("qiskit_optimization")
    expected_returns = pd.Series({"AAA": 0.10, "BBB": 0.08, "CCC": 0.05})
    covariance = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.03, 0.01], [0.00, 0.01, 0.02]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    problem = build_portfolio_problem(expected_returns, covariance, budget=2)

    assert problem.get_num_binary_vars() == 3
    assert [variable.name for variable in problem.variables] == ["AAA", "BBB", "CCC"]
    assert problem.linear_constraints[0].name == "budget"
    assert problem.linear_constraints[0].rhs == 2


def test_problem_rejects_non_symmetric_covariance() -> None:
    expected_returns = pd.Series({"AAA": 0.10, "BBB": 0.08})
    covariance = pd.DataFrame([[0.04, 0.01], [0.00, 0.03]], index=expected_returns.index, columns=expected_returns.index)

    with pytest.raises(ValueError, match="symmetric"):
        build_portfolio_problem(expected_returns, covariance, budget=1)

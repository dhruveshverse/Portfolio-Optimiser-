import numpy as np
from scipy.optimize import minimize
from scipy.stats import dirichlet


def calculate_averages(returns):
    """
    Calculate arithmetic mean of daily returns for each stock.
    Like Excel AVERAGE.
    """
    return returns.mean()


def calculate_expected_return(weights, averages):
    """
    Calculate expected return using SUMPRODUCT formula.
    expected_return(w) = sum over i of (w_i * average_i)
    """
    return np.sum(weights * averages)


def calculate_covariance(returns):
    """
    Calculate population covariance matrix (N x N).
    Uses np.cov with bias=True to divide by n (not n-1).
    Diagonal = population variance (like Excel VARP).
    """
    return np.cov(returns.T, bias=True)


def calculate_portfolio_risk(weights, cov_matrix):
    """
    Calculate portfolio risk following the exact sequence specified.
    
    Returns: (first_term, second_term, final_second_term, variance, risk)
    """
    # 6.1 first_term = sum over i of (w_i^2 * cov_ii)
    first_term = np.sum(weights**2 * np.diag(cov_matrix))
    
    # 6.2 second_term = sum over every pair i < j of (w_i * w_j * cov_ij)
    n = len(weights)
    second_term = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            second_term += weights[i] * weights[j] * cov_matrix[i, j]
    
    # 6.3 final_second_term = 2 * second_term
    final_second_term = 2 * second_term
    
    # 6.4 variance = first_term + final_second_term
    variance = first_term + final_second_term
    
    # Assert it equals w' @ Cov @ w within 1e-12
    matrix_result = weights @ cov_matrix @ weights
    if not np.isclose(variance, matrix_result, atol=1e-12):
        raise ValueError(f"Variance calculation mismatch: {variance} vs {matrix_result}")
    
    # 6.5 risk = sqrt(variance)
    risk = np.sqrt(variance)
    
    return first_term, second_term, final_second_term, variance, risk


def calculate_sharpe_ratio(expected_return, risk, risk_free_rate_percent):
    """
    Calculate return per risk (Sharpe-like ratio).
    rf_daily = (risk-free % / 100) / 252
    return_per_risk = (expected_return - rf_daily) / risk
    """
    rf_daily = (risk_free_rate_percent / 100) / 252
    if risk == 0:
        return "n/a", rf_daily
    return_per_risk = (expected_return - rf_daily) / risk
    return return_per_risk, rf_daily


def annualise_metrics(expected_return, risk, risk_free_rate_percent):
    """
    Annualise daily metrics.
    annual_return = expected_return * 252
    annual_risk = risk * sqrt(252)
    annual_ratio = (annual_return - risk-free decimal) / annual_risk
    """
    annual_return = expected_return * 252
    annual_risk = risk * np.sqrt(252)
    rf_decimal = risk_free_rate_percent / 100
    if annual_risk == 0:
        annual_ratio = "n/a"
    else:
        annual_ratio = (annual_return - rf_decimal) / annual_risk
    return annual_return, annual_risk, annual_ratio


def optimize_portfolio(averages, cov_matrix, max_weight):
    """
    Optimise portfolio to maximise return_per_risk (Sharpe ratio).
    Uses scipy.optimize.minimize with SLSQP.
    Runs from 50 random starting points plus equal-weight start.
    """
    n = len(averages)
    
    # Objective: minimise negative return_per_risk (maximise return_per_risk)
    def objective(weights):
        expected_return = calculate_expected_return(weights, averages)
        _, _, _, _, risk = calculate_portfolio_risk(weights, cov_matrix)
        if risk == 0:
            return -np.inf
        return -expected_return / risk
    
    # Constraints
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # sum of weights = 1
    )
    
    # Bounds: each weight between 0 and max_weight
    bounds = [(0, max_weight) for _ in range(n)]
    
    # Run from 50 random starting points plus equal-weight
    best_result = None
    best_objective = -np.inf
    
    np.random.seed(42)
    
    # Equal-weight start
    equal_weights = np.ones(n) / n
    result = minimize(
        objective,
        equal_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        tol=1e-12,
        options={'maxiter': 1000}
    )
    if result.success and result.fun > best_objective:
        best_result = result
        best_objective = result.fun
    
    # 50 random Dirichlet starting points
    for _ in range(50):
        random_weights = dirichlet.rvs(np.ones(n), size=1)[0]
        result = minimize(
            objective,
            random_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            tol=1e-12,
            options={'maxiter': 1000}
        )
        if result.success and result.fun > best_objective:
            best_result = result
            best_objective = result.fun
    
    if best_result is None:
        raise ValueError("Optimisation failed: no successful runs")
    
    weights = best_result.x
    
    # Round weights below 0.0005 to 0 and renormalise
    weights[weights < 0.0005] = 0
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    
    return weights

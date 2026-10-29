import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import portfolio


# Load Excel test data
EXCEL_PATH = "C:/Users/IT Tech/Downloads/Actual eg for portfolio optimisation.xlsx"


@pytest.fixture
def test_returns():
    """Load returns from Excel (rows 2 onward, columns B to F, 246 rows, no header)."""
    df = pd.read_excel(EXCEL_PATH, header=None)
    # Columns B to F are indices 1 to 5 (0-indexed)
    # Rows 2 onward means skip first row (index 0)
    returns_data = df.iloc[1:, 1:6]  # Get the data as DataFrame first
    # Convert to numeric, coercing errors to NaN
    returns_data = returns_data.apply(pd.to_numeric, errors='coerce')
    # Drop rows with any NaN values
    returns_data = returns_data.dropna()
    return returns_data.values


@pytest.fixture
def test_returns_df(test_returns):
    """Convert returns to DataFrame with stock names."""
    stock_names = ["HDFC Bank", "TCS", "Sun Pharma", "ITC", "L&T"]
    return pd.DataFrame(test_returns, columns=stock_names)


def test_averages(test_returns_df):
    """Test 9.1: Averages calculation."""
    averages = portfolio.calculate_averages(test_returns_df)
    
    expected = np.array([
        0.0016692185,
        0.0010637727,
        -0.0000235501,
        0.0020803608,
        0.0002986301
    ])
    
    np.testing.assert_allclose(averages.values, expected, atol=1e-9)


def test_covariance_diagonal(test_returns_df):
    """Test 9.2: Covariance diagonal elements."""
    cov_matrix = portfolio.calculate_covariance(test_returns_df)
    
    expected_diagonal = np.array([
        2.00533207e-04,
        3.60325530e-04,
        1.47134886e-04,
        2.14111489e-04,
        2.69130836e-04
    ])
    
    actual_diagonal = np.diag(cov_matrix)
    np.testing.assert_allclose(actual_diagonal, expected_diagonal, atol=1e-9)


def test_covariance_off_diagonal(test_returns_df):
    """Test 9.3: Specific covariance values."""
    cov_matrix = portfolio.calculate_covariance(test_returns_df)
    
    # TCS-HDFC: TCS is index 1, HDFC is index 0
    tcs_hdfc = cov_matrix[1, 0]
    expected_tcs_hdfc = 5.35227587e-05
    np.testing.assert_allclose(tcs_hdfc, expected_tcs_hdfc, atol=1e-9)
    
    # L&T-HDFC: L&T is index 4, HDFC is index 0
    lt_hdfc = cov_matrix[4, 0]
    expected_lt_hdfc = 1.17018140e-04
    np.testing.assert_allclose(lt_hdfc, expected_lt_hdfc, atol=1e-9)


def test_equal_weights_portfolio(test_returns_df):
    """Test 9.4: Equal weights portfolio."""
    averages = portfolio.calculate_averages(test_returns_df)
    cov_matrix = portfolio.calculate_covariance(test_returns_df)
    
    # Equal weights
    n = len(averages)
    weights = np.ones(n) / n
    
    expected_return = portfolio.calculate_expected_return(weights, averages)
    expected_return_value = 0.0010176864
    np.testing.assert_allclose(expected_return, expected_return_value, atol=1e-9)
    
    first_term, second_term, final_second_term, variance, risk = portfolio.calculate_portfolio_risk(weights, cov_matrix)
    expected_risk = 0.0095405889
    np.testing.assert_allclose(risk, expected_risk, atol=1e-9)
    
    # rf = 0
    return_per_risk, _ = portfolio.calculate_sharpe_ratio(expected_return, risk, 0)
    expected_return_per_risk = 0.1066691
    np.testing.assert_allclose(return_per_risk, expected_return_per_risk, atol=1e-6)


def test_optimised_portfolio(test_returns_df):
    """Test 9.5: Optimised portfolio with rf = 0, no max weight constraint."""
    averages = portfolio.calculate_averages(test_returns_df)
    cov_matrix = portfolio.calculate_covariance(test_returns_df)
    
    # Optimise with no max weight constraint (max_weight = 1.0)
    opt_weights = portfolio.optimize_portfolio(averages, cov_matrix, max_weight=1.0)
    
    # Check weights (tolerance 0.005)
    expected_weights = np.array([0.4011, 0.0582, 0, 0.5407, 0])
    np.testing.assert_allclose(opt_weights, expected_weights, atol=0.005)
    
    # Check expected return (tolerance 1e-5)
    opt_expected = portfolio.calculate_expected_return(opt_weights, averages)
    expected_return = 0.0018563
    np.testing.assert_allclose(opt_expected, expected_return, atol=1e-5)
    
    # Check risk (tolerance 1e-5)
    _, _, _, _, opt_risk = portfolio.calculate_portfolio_risk(opt_weights, cov_matrix)
    expected_risk = 0.0112052
    np.testing.assert_allclose(opt_risk, expected_risk, atol=1e-5)
    
    # Check return_per_risk (tolerance 1e-4)
    return_per_risk, _ = portfolio.calculate_sharpe_ratio(opt_expected, opt_risk, 0)
    expected_return_per_risk = 0.165662
    np.testing.assert_allclose(return_per_risk, expected_return_per_risk, atol=1e-4)


def test_return_formula():
    """Test 9.6: Return formula with closes 1000, 1020, 1000.8, 1010."""
    closes = np.array([1000, 1020, 1000.8, 1010])
    
    # Calculate returns: (close(t) - close(t-1)) / close(t-1)
    expected_returns = np.array([
        (1020 - 1000) / 1000,      # 0.02
        (1000.8 - 1020) / 1020,   # -0.0188235294...
        (1010 - 1000.8) / 1000.8  # 0.0091916367...
    ])
    
    expected_returns_rounded = np.array([0.02, -0.018824, 0.009192])
    
    np.testing.assert_allclose(expected_returns, expected_returns_rounded, atol=1e-6)


def test_risk_free_rate_adjustment(test_returns_df):
    """Test 9.7: Risk-free rate adjustment."""
    averages = portfolio.calculate_averages(test_returns_df)
    cov_matrix = portfolio.calculate_covariance(test_returns_df)
    
    # Equal weights
    n = len(averages)
    weights = np.ones(n) / n
    
    expected_return = portfolio.calculate_expected_return(weights, averages)
    _, _, _, _, risk = portfolio.calculate_portfolio_risk(weights, cov_matrix)
    
    # With rf = 6.5%
    rf_percent = 6.5
    return_per_risk, rf_daily = portfolio.calculate_sharpe_ratio(expected_return, risk, rf_percent)
    
    # Check rf_daily = 0.065/252
    expected_rf_daily = 0.065 / 252
    np.testing.assert_allclose(rf_daily, expected_rf_daily, atol=1e-9)
    
    # Check return_per_risk = (expected_return - rf_daily) / risk
    expected_return_per_risk = (expected_return - rf_daily) / risk
    np.testing.assert_allclose(return_per_risk, expected_return_per_risk, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

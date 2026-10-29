# Portfolio Optimiser

A Streamlit web application for portfolio optimisation using Modern Portfolio Theory. This tool helps investors analyse stock portfolios, calculate risk-return metrics, and find optimal weight allocations.

## Features

- **Stock Selection**: Choose from NSE-listed companies (2-25 stocks)
- **Flexible Tenure**: 1 year, 3 years, 5 years, or custom date range
- **Manual Weights**: Set custom portfolio weights or use optimisation
- **Risk Analysis**: Calculate portfolio risk using covariance matrices
- **Optimisation**: Maximise return/risk ratio using scipy optimisation
- **Visualisations**: Correlation heatmaps, efficient frontier charts, pie charts
- **Export**: Download results as Excel files

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd portfolio-optimiser
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Formulas

### Returns Calculation
Daily returns use simple percentage change:
```
daily_return(t) = (close(t) - close(t-1)) / close(t-1)
```

### Averages and Expected Return
- **Average**: Arithmetic mean of daily returns (Excel AVERAGE)
- **Expected Return**: Weighted sum (Excel SUMPRODUCT)
```
expected_return(w) = Σ (w_i × average_i)
```

### Covariance Matrix
Population covariance (divides by n, not n-1):
```
cov_ij = Σ ((r_i(t) - avg_i) × (r_j(t) - avg_j)) / n
```

### Portfolio Risk
Calculated in sequence:
1. First term: Σ (w_i² × variance_i)
2. Second term: Σ (w_i × w_j × cov_ij) for i < j
3. Final second term: 2 × second term
4. Variance: first term + final second term
5. Risk: √variance
6. Return per risk: (expected_return - rf_daily) / risk

### Optimisation
- **Objective**: Maximise return/risk ratio
- **Constraints**: Weights sum to 1, 0 ≤ weight ≤ max_weight
- **Method**: scipy.optimize.minimize (SLSQP)
- **Starting points**: 50 random Dirichlet + equal-weight

## Project Structure

```
portfolio-optimiser/
├── app.py              # Streamlit UI
├── portfolio.py        # Mathematical calculations
├── data.py             # Data download and processing
├── nse_companies.csv   # NSE company list
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── tests/
    └── test_portfolio.py  # Pytest tests
```

## Testing

Run the test suite:
```bash
pytest tests/test_portfolio.py -v
```

Tests verify calculations against ground truth data from the provided Excel file.

## Screenshots

(Add screenshots of the app interface here)

## Deployment

The app is deployed on Streamlit Community Cloud at:
[Public Link - to be added after deployment]

## Disclaimer

**Educational use only. Not investment advice. Past returns do not guarantee future returns.**

## License

MIT License

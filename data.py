import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache


@lru_cache(maxsize=None)
def download_stock_data(tickers, start_date, end_date):
    """
    Download stock data from yfinance for given tickers and date range.
    Returns a DataFrame with dates as rows and tickers as columns.
    """
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    if 'Close' in data:
        return data['Close']
    elif len(data.columns) == 1:
        return data
    else:
        raise ValueError("Unexpected data format from yfinance")


def prepare_data(selected_stocks, start_date, end_date):
    """
    Download and prepare stock data for selected stocks.
    """
    # Append .NS suffix for NSE stocks
    tickers = [f"{stock}.NS" for stock in selected_stocks]
    
    # Download data
    prices = download_stock_data(tuple(tickers), start_date, end_date)
    
    # Check if any ticker returned no data
    missing_data = []
    for i, ticker in enumerate(tickers):
        if ticker not in prices.columns or prices[ticker].isna().all():
            missing_data.append(selected_stocks[i])
    
    if missing_data:
        raise ValueError(f"No data available for: {', '.join(missing_data)}")
    
    # Drop dates where ANY stock has no price (inner join on dates)
    prices = prices.dropna()
    
    # Sort dates OLDEST to NEWEST
    prices = prices.sort_index()
    
    # Check minimum rows
    if len(prices) < 30:
        raise ValueError(f"Insufficient data: only {len(prices)} rows available (minimum 30 required)")
    
    return prices


def calculate_returns(prices):
    """
    Calculate daily returns using simple percentage change.
    daily_return(t) = (close(t) - close(t-1)) / close(t-1)
    """
    returns = prices.pct_change().dropna()
    return returns

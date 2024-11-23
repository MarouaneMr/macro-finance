import pandas as pd

def load_price_data(filepath, tickers):
    """
    Load price data from a CSV file, filtering only the selected tickers.
    """
    prices = pd.read_csv(filepath, parse_dates=['Date'])
    prices.set_index('Date', inplace=True)
    prices = prices[tickers]  # Filter for selected tickers
    return prices

def calculate_returns(prices_df):
    """
    Calculate monthly returns from price data.
    """
    returns = prices_df.pct_change().dropna()
    return returns

def filter_stocks_from_metrics(metrics_file, top_n=30, bottom_n=10):
    """
    Filter the top N and bottom N stocks based on return-to-risk ratio.
    """
    metrics = pd.read_csv(metrics_file)
    metrics['Return-to-Risk'] = metrics['Average Return'] / metrics['Risk']  # Calculate return-to-risk ratio

    # Sort by return-to-risk ratio
    top_stocks = metrics.sort_values('Return-to-Risk', ascending=False).head(top_n)['Symbol'].tolist()
    bottom_stocks = metrics.sort_values('Return-to-Risk', ascending=True).head(bottom_n)['Symbol'].tolist()

    return top_stocks, bottom_stocks

import pandas as pd

def load_price_data(filepath):
    """
    Load price data from a CSV file.
    """
    prices = pd.read_csv(filepath, parse_dates=['Date'])
    prices.set_index('Date', inplace=True)
    return prices

def calculate_returns(prices_df):
    """
    Calculate monthly returns from price data.
    """
    returns = prices_df.pct_change().dropna()
    return returns

def filter_tickers_by_metadata(prices_df, metadata_file, filter_col, filter_val=None):
    """
    Filter tickers based on metadata.
    If filter_val is None, return all tickers in the metadata.
    """
    metadata = pd.read_csv(metadata_file)
    selected_tickers = metadata[filter_col].tolist()  # Extract tickers from the column
    filtered_prices = prices_df[prices_df.columns.intersection(selected_tickers)]
    return filtered_prices
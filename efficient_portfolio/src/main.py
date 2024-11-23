import pandas as pd
import numpy as np
from data_processing import calculate_returns, load_price_data, filter_stocks_from_metrics
from portfolio_optimization import (
    optimize_efficient_frontier,
    calculate_portfolio_metrics,
)
from visualize import plot_efficient_frontier


def save_portfolio_results(label, weights, tickers, portfolio_return, portfolio_risk, sharpe_ratio, folder="results"):
    """
    Save portfolio results as a text file.
    """
    selected_assets = [(ticker, weight) for ticker, weight in zip(tickers, weights) if abs(weight) > 0.01]  # Non-zero weights
    result_lines = [
        f"Portfolio: {label}",
        f"Expected Return: {portfolio_return:.2%}",
        f"Risk (Standard Deviation): {portfolio_risk:.2%}",
        f"Sharpe Ratio: {sharpe_ratio:.2f}",
        "Selected Stocks and Weights:",
        "\n".join([f"{ticker}: {weight:.2%}" for ticker, weight in selected_assets]),
        f"Total Assets Held: {len(selected_assets)}"
    ]

    with open(f"{folder}/{label}_portfolio.txt", "w") as f:
        f.write("\n".join(result_lines))
    print(f"{label} portfolio saved in {folder}/{label}_portfolio.txt")


# Load metrics data and filter stocks
metrics_file = 'efficient_portfolio/data/metrics.csv'  # File containing return-to-risk metrics
prices_file = 'efficient_portfolio/data/monthly_prices.csv'

# Step 1: Filter stocks based on return-to-risk ratio
top_stocks, bottom_stocks = filter_stocks_from_metrics(metrics_file, top_n=30, bottom_n=10)

# Step 2: Load price data for the selected stocks
selected_tickers = top_stocks + bottom_stocks
prices = load_price_data(prices_file, selected_tickers)

# Step 3: Calculate returns for the filtered stocks
returns = calculate_returns(prices)

# Step 4: Compute mean returns and covariance matrix
mean_returns = returns.mean().to_numpy()
covariance_matrix = returns.cov().to_numpy()
tickers = returns.columns.tolist()

# Generate Portfolios
def generate_portfolios():
    """
    Generate portfolios for each constraint case.
    """
    # Unconstrained Portfolio
    weights_unconstrained = optimize_efficient_frontier(mean_returns, covariance_matrix, max(mean_returns), weight_bounds=None)
    return_unconstrained, risk_unconstrained, sharpe_unconstrained = calculate_portfolio_metrics(
        weights_unconstrained, mean_returns, covariance_matrix
    )
    save_portfolio_results(
        "Unconstrained",
        weights_unconstrained,
        tickers,
        return_unconstrained,
        risk_unconstrained,
        sharpe_unconstrained
    )

    # Long-Only Portfolio
    weights_long_only = optimize_efficient_frontier(mean_returns, covariance_matrix, max(mean_returns), weight_bounds=[(0, None)] * len(tickers))
    return_long_only, risk_long_only, sharpe_long_only = calculate_portfolio_metrics(
        weights_long_only, mean_returns, covariance_matrix
    )
    save_portfolio_results(
        "Long-Only",
        weights_long_only,
        tickers,
        return_long_only,
        risk_long_only,
        sharpe_long_only
    )

    # Weight-Bounded Portfolio (0 <= w_i <= 0.05)
    weights_bounded = optimize_efficient_frontier(mean_returns, covariance_matrix, max(mean_returns), weight_bounds=[(0, 0.05)] * len(tickers))
    return_bounded, risk_bounded, sharpe_bounded = calculate_portfolio_metrics(
        weights_bounded, mean_returns, covariance_matrix
    )
    save_portfolio_results(
        "Weight-Bounded",
        weights_bounded,
        tickers,
        return_bounded,
        risk_bounded,
        sharpe_bounded
    )

generate_portfolios()

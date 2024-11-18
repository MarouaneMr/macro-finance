import pandas as pd
import numpy as np
from data_processing import load_price_data, calculate_returns, filter_tickers_by_metadata
from portfolio_optimization import (
    calculate_portfolio_metrics,
    optimize_unconstrained,
    optimize_long_only,
    optimize_weight_bounds,
    get_portfolio_composition,
    save_portfolio_composition,
)
from visualize import plot_efficient_frontier, plot_portfolio_composition

def generate_portfolio_summary(portfolio_name, portfolio_return, portfolio_risk, sharpe_ratio, weights, tickers):
    """
    Generate a human-readable summary for a portfolio.
    """
    composition = get_portfolio_composition(weights, tickers)
    sorted_composition = sorted(composition.items(), key=lambda x: -x[1])  # Sort by weight descending
    top_assets = [f"{ticker}: {weight:.2%}" for ticker, weight in sorted_composition[:5]]  # Top 5 holdings

    summary = f"""
    {portfolio_name} Portfolio Summary:
    ----------------------------------------
    - Expected Return: {portfolio_return:.2%}
    - Risk (Standard Deviation): {portfolio_risk:.2%}
    - Sharpe Ratio: {sharpe_ratio:.2f} (if applicable)
    - Top 5 Holdings: {', '.join(top_assets)}
    - Total Assets: {len(tickers)} assets
    """
    print(summary)
    return summary

# Load and preprocess data
prices = load_price_data('efficient_portfolio/data/monthly_prices.csv')

# Filter prices for S&P 500 tickers
filtered_prices = filter_tickers_by_metadata(prices, 'efficient_portfolio/data/tickers_sp500.csv', 'Symbol', None)
returns = calculate_returns(filtered_prices)

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
covariance_matrix = returns.cov()

# Optimize portfolios
weights_unconstrained = optimize_unconstrained(mean_returns, covariance_matrix)
weights_long_only = optimize_long_only(mean_returns, covariance_matrix)
weights_weight_bounds = optimize_weight_bounds(mean_returns, covariance_matrix, lower=0, upper=0.05)

# Calculate metrics for each portfolio
metrics_unconstrained = calculate_portfolio_metrics(weights_unconstrained, mean_returns, covariance_matrix)
metrics_long_only = calculate_portfolio_metrics(weights_long_only, mean_returns, covariance_matrix)
metrics_weight_bounds = calculate_portfolio_metrics(weights_weight_bounds, mean_returns, covariance_matrix)

# Save portfolio compositions
tickers = returns.columns.tolist()
save_portfolio_composition('efficient_portfolio/results/unconstrained_portfolio.csv', 'Unconstrained', weights_unconstrained, tickers)
save_portfolio_composition('efficient_portfolio/results/long_only_portfolio.csv', 'Long-Only', weights_long_only, tickers)
save_portfolio_composition('efficient_portfolio/results/weight_bounds_portfolio.csv', 'Weight-Bounded', weights_weight_bounds, tickers)

# Generate and print summaries
summary_unconstrained = generate_portfolio_summary(
    "Unconstrained", *metrics_unconstrained, weights_unconstrained, tickers
)
summary_long_only = generate_portfolio_summary(
    "Long-Only", *metrics_long_only, weights_long_only, tickers
)
summary_weight_bounds = generate_portfolio_summary(
    "Weight-Bounded", *metrics_weight_bounds, weights_weight_bounds, tickers
)

# Save summaries to text files
with open('efficient_portfolio/results/portfolio_summaries.txt', 'w') as f:
    f.write(summary_unconstrained)
    f.write(summary_long_only)
    f.write(summary_weight_bounds)

# Visualize results
plot_efficient_frontier(pd.DataFrame({
    'Risk': [metrics_unconstrained[1], metrics_long_only[1], metrics_weight_bounds[1]],
    'Return': [metrics_unconstrained[0], metrics_long_only[0], metrics_weight_bounds[0]],
}), "Efficient Frontiers for S&P 500")

plot_portfolio_composition(weights_unconstrained, tickers, "Unconstrained Portfolio (S&P 500)")
plot_portfolio_composition(weights_long_only, tickers, "Long-Only Portfolio (S&P 500)")
plot_portfolio_composition(weights_weight_bounds, tickers, "Weight-Bounded Portfolio (S&P 500)")

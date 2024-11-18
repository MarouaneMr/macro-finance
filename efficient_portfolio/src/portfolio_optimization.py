from scipy.optimize import minimize
import numpy as np
import pandas as pd

def calculate_portfolio_metrics(weights, mean_returns, covariance_matrix, risk_free_rate=0.01):
    """
    Calculate portfolio metrics: expected return, variance, and Sharpe ratio.
    """
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    portfolio_std_dev = np.sqrt(portfolio_variance)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return portfolio_return, portfolio_std_dev, sharpe_ratio

def optimize_unconstrained(mean_returns, covariance_matrix):
    """
    Optimize portfolio without constraints.
    """
    num_assets = len(mean_returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(None, None) for _ in range(num_assets)]
    initial_guess = num_assets * [1.0 / num_assets]

    result = minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def optimize_long_only(mean_returns, covariance_matrix):
    """
    Optimize portfolio with long-only constraint.
    """
    num_assets = len(mean_returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(0, None) for _ in range(num_assets)]
    initial_guess = num_assets * [1.0 / num_assets]

    result = minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def optimize_weight_bounds(mean_returns, covariance_matrix, lower=0, upper=0.05):
    """
    Optimize portfolio with weight bounds.
    """
    num_assets = len(mean_returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(lower, upper) for _ in range(num_assets)]
    initial_guess = num_assets * [1.0 / num_assets]

    result = minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def get_portfolio_composition(weights, tickers):
    """
    Generate a dictionary mapping tickers to weights.
    """
    return dict(zip(tickers, weights))

def save_portfolio_composition(filepath, portfolio_name, weights, tickers):
    """
    Save portfolio composition to a CSV file.
    """
    composition = pd.DataFrame({'Ticker': tickers, 'Weight': weights})
    composition.to_csv(filepath, index=False)
    print(f"{portfolio_name} composition saved to {filepath}")

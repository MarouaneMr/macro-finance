from scipy.optimize import minimize
import numpy as np

def calculate_portfolio_metrics(weights, mean_returns, covariance_matrix, risk_free_rate=0.01):
    """
    Calculate portfolio metrics: return, risk, and Sharpe ratio.
    """
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    portfolio_risk = np.sqrt(portfolio_variance)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
    return portfolio_return, portfolio_risk, sharpe_ratio

def optimize_efficient_frontier(mean_returns, covariance_matrix, target_return, weight_bounds=None):
    """
    Optimize portfolio for a target return by minimizing risk.
    """
    num_assets = len(mean_returns)

    # Objective: Minimize portfolio variance (risk)
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(covariance_matrix, weights))

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # Weights sum to 1
        {'type': 'eq', 'fun': lambda weights: np.dot(weights, mean_returns) - target_return},  # Target return
    ]

    # Bounds for weights
    bounds = weight_bounds if weight_bounds else [(None, None)] * num_assets

    # Initial guess
    initial_guess = np.full(num_assets, 1.0 / num_assets)

    # Solve optimization
    result = minimize(portfolio_variance, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

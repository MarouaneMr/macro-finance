import matplotlib.pyplot as plt
import numpy as np

def plot_efficient_frontiers(portfolio_results):
    """
    Plot efficient frontiers for all portfolios.
    """
    plt.figure(figsize=(12, 8))
    for label, result in portfolio_results.items():
        plt.scatter(result["risk"], result["return"], label=f"{label} Portfolio", s=100)
    plt.xlabel("Risk (Standard Deviation)", fontsize=14)
    plt.ylabel("Expected Return", fontsize=14)
    plt.title("Efficient Frontiers", fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig("results/efficient_frontiers.png")
    plt.show()

def plot_portfolio_composition(weights, tickers, label):
    """
    Plot portfolio composition as a pie chart.
    Filters out negative weights and handles short positions separately.
    """
    non_zero_indices = np.where(np.abs(weights) > 0.01)[0]  # Filter out small weights
    non_zero_weights = weights[non_zero_indices]
    non_zero_tickers = np.array(tickers)[non_zero_indices]

    # Separate long and short positions
    long_indices = np.where(non_zero_weights > 0)[0]
    long_weights = non_zero_weights[long_indices]
    long_tickers = non_zero_tickers[long_indices]

    if len(long_weights) > 0:
        # Plot pie chart for long positions only
        plt.figure(figsize=(10, 8))
        plt.pie(long_weights, labels=long_tickers, autopct='%1.1f%%', startangle=140)
        plt.title(f"{label} Portfolio Composition (Long Positions)", fontsize=16)
        plt.savefig(f"results/{label}_portfolio_composition.png")
        plt.show()
    else:
        print(f"No long positions to visualize in {label} portfolio.")

def plot_risk_return_summary(portfolio_results):
    """
    Plot risk-return summary for all portfolios as a bar chart.
    """
    labels = list(portfolio_results.keys())
    returns = [result["return"] for result in portfolio_results.values()]
    risks = [result["risk"] for result in portfolio_results.values()]
    sharpes = [result["sharpe"] for result in portfolio_results.values()]

    x = np.arange(len(labels))  # Label locations
    width = 0.25  # Bar width

    plt.figure(figsize=(14, 8))
    plt.bar(x - width, returns, width, label='Return', color='green')
    plt.bar(x, risks, width, label='Risk', color='red')
    plt.bar(x + width, sharpes, width, label='Sharpe Ratio', color='blue')

    plt.xlabel("Portfolio Type", fontsize=14)
    plt.ylabel("Metrics", fontsize=14)
    plt.title("Portfolio Risk-Return Summary", fontsize=16)
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y')
    plt.savefig("results/risk_return_summary.png")
    plt.show()

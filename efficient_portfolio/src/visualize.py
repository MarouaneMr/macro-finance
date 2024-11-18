import matplotlib.pyplot as plt
import seaborn as sns

def plot_efficient_frontier(frontier_data, title):
    """
    Plot the efficient frontier with risk vs. return.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Risk', y='Return', data=frontier_data, label='Efficient Frontier')
    plt.xlabel('Portfolio Risk (Std Dev)')
    plt.ylabel('Portfolio Return')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_portfolio_composition(weights, tickers, title):
    """
    Plot portfolio composition as a bar chart.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(tickers, weights, color='skyblue')
    plt.xlabel('Assets')
    plt.ylabel('Portfolio Weights')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()
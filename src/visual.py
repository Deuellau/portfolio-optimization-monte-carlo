# visual.py
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# -----------------------------
# Efficient Frontier
# -----------------------------
def plot_efficient_frontier(all_portfolios_df, best_results):
    plt.style.use('seaborn-v0_8-white')
    plt.figure(figsize=(10,6))

    scatter = plt.scatter(
        all_portfolios_df["volatility"], 
        all_portfolios_df["annualized_return"],
        c=all_portfolios_df["sharpe"], 
        cmap="viridis", 
        s=3,
        alpha=0.8
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Sharpe Ratio", fontsize=12, weight='bold')

    # Highlight max Sharpe
    plt.scatter(
        best_results[1], best_results[0], 
        color="#D62728", s=200, marker='*', label='Max Sharpe'
    )

    plt.xlabel("Volatility", fontsize=12, weight='bold')
    plt.ylabel("Annualized Return", fontsize=12, weight='bold')
    plt.title("Efficient Frontier", fontsize=16, weight='bold', pad=15)
    plt.legend(frameon=False, fontsize=12)
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Cumulative Growth
# -----------------------------
def plot_cumulative(portfolio_series, benchmark_series, portfolio_label="Portfolio", benchmark_label="Benchmark"):
    plt.style.use('seaborn-v0_8-white')
    plt.figure(figsize=(12,6))

    plt.plot(portfolio_series, label=portfolio_label, color="#007BC6", linewidth=2)
    plt.plot(benchmark_series, label=benchmark_label, color="#153C78", linewidth=2)

    plt.xlabel("Year", fontsize=12, weight='bold')
    plt.ylabel("Cumulative Value", fontsize=12, weight='bold')
    plt.title(f"{portfolio_label} vs {benchmark_label}", fontsize=16, weight='bold', pad=15)
    plt.legend(frameon=False, fontsize=12)

    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)
    for spine in ["top","right","left"]:
        plt.gca().spines[spine].set_visible(False)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.tight_layout()
    plt.show()

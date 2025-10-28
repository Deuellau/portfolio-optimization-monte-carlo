# main.py
import pandas as pd
import numpy as np
from config import TICKERS, RF_RATE
from compute import load_data, monte_carlo_portfolio, rolling_monte_carlo, compute_metrics, median_portfolio_quarterly
from visual import plot_efficient_frontier, plot_cumulative

# -----------------------------
# 1. Load data
# -----------------------------
data = load_data()
returns = data.pct_change().dropna()

# -----------------------------
# 2. Monte Carlo Simulation
# -----------------------------
all_portfolios, best_subset, best_weights, best_results = monte_carlo_portfolio(returns)

# -----------------------------
# Map best subset indices to tickers and print results
# -----------------------------
best_tickers = [TICKERS[i] for i in best_subset]

print("Optimal Portfolio Weights (Quarterly Rebalanced Subset):")
for t, w in zip(best_tickers, best_weights):
    print(f"{t}: {w:.2%}")

print(f"\nExpected Annualized Return: {best_results[0]:.2%}")
print(f"Volatility: {best_results[1]:.2%}")
print(f"Sharpe Ratio: {best_results[2]:.3f}")

# Convert to DataFrame for plotting
all_portfolios_df = pd.DataFrame(all_portfolios)

# Plot efficient frontier
plot_efficient_frontier(all_portfolios_df, best_results)

# Extract best portfolio series
for p in all_portfolios:
    if len(p['weights']) == len(best_weights) and np.allclose(p['weights'], best_weights) and p['subset'] == [TICKERS[i] for i in best_subset]:
        best_portfolio_series = pd.Series(p['series'], index=returns.index[:len(p['series'])])
        break

spy_series = (1 + returns['SPY']).cumprod()

# Plot cumulative growth
plot_cumulative(best_portfolio_series, spy_series, "Optimal Portfolio", "SPY")

# -----------------------------
# 3. Rolling Monte Carlo
# -----------------------------
rolling_series, rolling_weights = rolling_monte_carlo(returns)
rolling_weights_df = pd.DataFrame(rolling_weights)
print(rolling_weights_df)

# -----------------------------
# 4. Median-Weight Portfolio
# -----------------------------
weights_rows = []
for rw in rolling_weights:
    row = pd.Series(0.0, index=TICKERS)
    for asset, w in zip(rw['subset'], rw['weights']):
        row[asset] = w
    weights_rows.append(row)

weights_df = pd.DataFrame(weights_rows)
median_weights = weights_df.median(axis=0)
median_weights /= median_weights.sum()
print("\nMedian Portfolio Weights (normalized):")
print(median_weights.sort_values(ascending=False))


# -----------------------------
# Quarterly Rebalanced Median Portfolio
# -----------------------------
start_date, end_date, cumulative_portfolio, cumulative_spy = median_portfolio_quarterly(returns, median_weights)

# Compute metrics
port_ann, port_vol, port_sharpe = compute_metrics(cumulative_portfolio)
spy_ann, spy_vol, spy_sharpe = compute_metrics(cumulative_spy)

print(f"\nPerformance of Median-Weight Portfolio (Quarterly Rebalanced, {start_date} to {end_date}):")
print(f"Annualized Return: {port_ann:.2%}")
print(f"Volatility:        {port_vol:.2%}")
print(f"Sharpe Ratio:      {port_sharpe:.2f}")

print("\nPerformance of SPY (Same Period):")
print(f"Annualized Return: {spy_ann:.2%}")
print(f"Volatility:        {spy_vol:.2%}")
print(f"Sharpe Ratio:      {spy_sharpe:.2f}")

# Plot cumulative growth
plot_cumulative(
    cumulative_portfolio,
    cumulative_spy,
    "Median-Weight Portfolio (Quarterly Rebalanced)",
    "SPY"
)

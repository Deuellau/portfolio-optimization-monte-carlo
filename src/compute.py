# compute.py
import pandas as pd
import numpy as np
from itertools import combinations
from config import TICKERS, RF_RATE, NUM_PORTFOLIOS, WINDOW_SIZE, STEP_SIZE, DATA_PATH, MED_START_DATE, MED_END_DATE

# -----------------------------
# Load price data
# -----------------------------
def load_data(path=DATA_PATH):
    print("\n[Stage] Loading price data...")
    data = pd.read_csv(path)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    print(f"[Stage] Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    return data

# -----------------------------
# Portfolio metrics
# -----------------------------
def compute_metrics(series, rf_rate=RF_RATE):
    total_days = len(series)
    total_return = series[-1] / series[0] - 1
    annualized_return = (1 + total_return) ** (252 / total_days) - 1
    vol = np.std(np.diff(np.log(series))) * np.sqrt(252)
    sharpe = (annualized_return - rf_rate) / vol
    return annualized_return, vol, sharpe

# -----------------------------
# Quarterly rebalanced Monte Carlo
# -----------------------------
def monte_carlo_portfolio(returns_df, tickers=TICKERS, rf_rate=RF_RATE):
    print("[Stage] Starting Monte Carlo simulation for all ticker subsets...")
    returns_array = returns_df.values
    dates = returns_df.index
    quarter_ends = returns_df.resample("QE").apply(lambda x: x.index[-1]).index.values
    quarter_indices = np.searchsorted(dates.values, quarter_ends)

    best_sharpe = -np.inf
    best_subset = None
    best_weights = None
    best_results = None
    all_portfolios = []

    total_subsets = sum([len(list(combinations(range(len(tickers)), r))) for r in range(1, len(tickers)+1)])
    subset_counter = 0

    for r in range(1, len(tickers)+1):
        for subset in combinations(range(len(tickers)), r):
            subset_counter += 1
            print(f"[MC] Evaluating subset {subset_counter}/{total_subsets}: {[tickers[i] for i in subset]}")
            subset_returns = returns_array[:, subset]
            n_assets = subset_returns.shape[1]

            for _ in range(NUM_PORTFOLIOS):
                weights = np.random.random(n_assets)
                weights /= np.sum(weights)

                portfolio_value = 1.0
                portfolio_series = []

                start_idx = 0
                for q_end in quarter_indices:
                    chunk_returns = subset_returns[start_idx:q_end+1]
                    daily_growth = 1 + chunk_returns @ weights
                    cumulative = portfolio_value * np.cumprod(daily_growth)
                    portfolio_series.extend(cumulative)
                    portfolio_value = cumulative[-1]
                    start_idx = q_end + 1

                portfolio_series = np.array(portfolio_series)

                # Metrics
                total_days = len(portfolio_series)
                total_return = portfolio_series[-1] / portfolio_series[0] - 1
                annualized_return = (1 + total_return) ** (252 / total_days) - 1
                vol = np.std(np.diff(np.log(portfolio_series))) * np.sqrt(252)
                sharpe = (annualized_return - rf_rate) / vol

                all_portfolios.append({
                    "subset": [tickers[i] for i in subset],
                    "weights": weights,
                    "annualized_return": annualized_return,
                    "volatility": vol,
                    "sharpe": sharpe,
                    "series": portfolio_series
                })

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_subset = subset
                    best_weights = weights
                    best_results = (annualized_return, vol, sharpe)

    print("\n[Stage] Monte Carlo simulation complete.")
    return all_portfolios, best_subset, best_weights, best_results

# -----------------------------
# Rolling Monte Carlo
# -----------------------------
def rolling_monte_carlo(returns_df):
    print("\n[Stage] Starting Rolling Monte Carlo simulation...")
    tickers = returns_df.columns.tolist()
    rolling_portfolio_series = pd.Series(index=returns_df.index, dtype=float)
    rolling_weights = []

    total_windows = (len(returns_df) - WINDOW_SIZE) // STEP_SIZE
    window_counter = 0

    for start in range(0, len(returns_df) - WINDOW_SIZE, STEP_SIZE):
        window_counter += 1
        end = start + WINDOW_SIZE
        print(f"[Rolling] Processing window {window_counter}/{total_windows}: {returns_df.index[start]} to {returns_df.index[end-1]}")
        train_returns = returns_df.iloc[start:end]
        test_returns = returns_df.iloc[end:end+STEP_SIZE]

        if test_returns.empty:
            break

        # Find best portfolio on TRAIN
        best_sharpe = -np.inf
        best_subset = None
        best_weights = None
        returns_array = train_returns.values

        for r in range(1, len(tickers)+1):
            for subset in combinations(range(len(tickers)), r):
                subset_train = returns_array[:, list(subset)]
                n_assets = subset_train.shape[1]

                for _ in range(NUM_PORTFOLIOS):
                    weights = np.random.random(n_assets)
                    weights /= np.sum(weights)

                    mean_returns = subset_train.mean(axis=0)
                    port_ret = np.sum(weights * mean_returns) * 252
                    cov_matrix = np.cov(subset_train.T) * 252
                    cov_matrix = np.atleast_2d(cov_matrix)
                    port_vol = np.sqrt(weights @ cov_matrix @ weights)
                    sharpe = (port_ret - RF_RATE) / port_vol

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_subset = subset
                        best_weights = weights

        # Apply best weights to TEST
        subset_test = test_returns.iloc[:, list(best_subset)].values
        daily_growth = 1 + subset_test @ best_weights
        cumulative_growth = pd.Series(daily_growth).cumprod()

        # Stitch cumulative series
        if rolling_portfolio_series.dropna().empty:
            rolling_portfolio_series.loc[test_returns.index] = cumulative_growth.values
        else:
            last_val = rolling_portfolio_series.dropna().iloc[-1]
            rolling_portfolio_series.loc[test_returns.index] = last_val * cumulative_growth.values

        rolling_weights.append({
            "start": returns_df.index[start],
            "end": returns_df.index[end],
            "subset": [tickers[i] for i in best_subset],
            "weights": best_weights,
            "sharpe": best_sharpe
        })

    print("\n[Stage] Rolling Monte Carlo simulation complete.")
    return rolling_portfolio_series, rolling_weights


# -----------------------------
# Quarterly Rebalanced Median Portfolio
# -----------------------------
def median_portfolio_quarterly(returns_df, median_weights, start_date=MED_START_DATE, end_date=MED_END_DATE):
    
    print("\n[Stage] Computing quarterly rebalanced median-weight portfolio...")

    # --- Filter & prepare ---
    available_tickers = median_weights.index.intersection(returns_df.columns)
    median_weights = median_weights[available_tickers]
    portfolio_returns = returns_df[available_tickers].loc[start_date:end_date]
    spy_returns = returns_df["SPY"].loc[start_date:end_date]

    # Normalize weights
    median_weights = median_weights / median_weights.sum()

    # --- Quarterly rebalancing simulation ---
    dates = portfolio_returns.index
    weights = median_weights.values
    quarter_ends = portfolio_returns.resample("QE").apply(lambda x: x.index[-1]).index

    # Ensure last date is included as final rebalance
    if quarter_ends[-1] < portfolio_returns.index[-1]:
        quarter_ends = quarter_ends.append(pd.Index([portfolio_returns.index[-1]]))

    portfolio_value = 1.0
    current_values = portfolio_value * weights
    portfolio_series = []

    for date in dates:
        daily_returns = portfolio_returns.loc[date].values
        current_values *= (1 + daily_returns)
        portfolio_value = np.sum(current_values)
        portfolio_series.append(portfolio_value)

        # Rebalance at quarter end
        if date in quarter_ends:
            current_values = portfolio_value * weights

    cumulative_portfolio = pd.Series(portfolio_series, index=dates)
    cumulative_spy = (1 + spy_returns).cumprod()

    print("[Stage] Quarterly rebalanced median-weight portfolio computed successfully.")
    return start_date, end_date, cumulative_portfolio, cumulative_spy


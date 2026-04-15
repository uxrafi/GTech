
"""
testproject.py

Georgia Tech - CS 7646: Machine Learning for Trading
Project 6: Technical Indicators & Theoretically Optimal Strategy

This script serves as the main entry point to:
- Evaluate the Theoretically Optimal Strategy (TOS) based on foresight
- Simulate portfolio values using an improved market simulator
- Compare the TOS portfolio performance with a benchmark strategy (buy & hold)
- Plot and save a chart of normalized TOS and benchmark values
- Print portfolio statistics such as cumulative return, Sharpe ratio, etc.

Dependencies:
- TheoreticallyOptimalStrategy.py (for TOS logic)
- marketsimcode.py (for portfolio value computation)
- indicators.py (for visualization, optional)
- util.py (for data fetching)

This script also satisfies project requirements:
- Starts with $100,000
- Uses JPM stock over 01/01/2008–12/31/2009
- Uses 0 commission and 0 market impact
- Outputs charts and stats for the report

Author and study group information are provided at the end of the file.
"""

###################################
# STEP 1: Import necessary libraries and helper modules
# STEP 2: Define utility functions for statistics computation
# STEP 3: Run testproject() to evaluate TOS and benchmark
# STEP 4: Calculate stats and print performance comparison
# STEP 5: Plot normalized portfolio and benchmark
# STEP 6: Author and study group info
###################################

# STEP 1: Import necessary libraries and helper modules
import datetime as dt               # for date handling
import pandas as pd                 # for data manipulation
import numpy as np                  # for numerical operations
import matplotlib.pyplot as plt     # for plotting

import TheoreticallyOptimalStrategy as tos 
import marketsimcode as ms
import indicators  # Optional if indicator visualization is required

from util import get_data  # To fetch stock price data

# STEP 2: Define utility function for computing portfolio statistics
def compute_portfolio_stats(portvals):
    # Compute daily returns
    daily_rets = portvals.pct_change().dropna()

    # Compute cumulative return, average daily return, std dev, and Sharpe ratio
    cum_return = (portvals[-1] / portvals[0]) - 1
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sharpe_ratio = (avg_daily_ret / std_daily_ret) * np.sqrt(252)

    return cum_return, avg_daily_ret, std_daily_ret, sharpe_ratio

# STEP 3: Main function to evaluate the Theoretically Optimal Strategy
def testproject():
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    start_val = 100000

    # Get trades DataFrame from TheoreticallyOptimalStrategy
    trades = tos.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_val)

    # Compute portfolio values using marketsimcode
    portvals = ms.compute_portvals(trades, start_val=start_val, commission=0.0, impact=0.0)
    portvals = portvals["Portfolio Value"] if isinstance(portvals, pd.DataFrame) else portvals

    # Fetch benchmark prices for JPM and compute benchmark value (buy & hold 1000 shares)
    prices_all = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices_all[symbol].loc[start_date:end_date]
    benchmark_vals = prices * 1000
    benchmark_vals = benchmark_vals / benchmark_vals.iloc[0] * start_val  # Normalize to match starting value

    # Normalize TOS portfolio value
    norm_portvals = portvals / portvals.iloc[0] * start_val

    # STEP 4: Calculate performance statistics for TOS and benchmark
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(norm_portvals)
    cum_ret_b, avg_daily_ret_b, std_daily_ret_b, sharpe_ratio_b = compute_portfolio_stats(benchmark_vals)

    # Print portfolio statistics to 6 decimal places
    print(f"Date Range: {start_date.date()} to {end_date.date()}\n")
    print(f"Sharpe Ratio of TOS: {sharpe_ratio:.6f}")
    print(f"Sharpe Ratio of Benchmark: {sharpe_ratio_b:.6f}\n")
    print(f"Cumulative Return of TOS: {cum_ret:.6f}")
    print(f"Cumulative Return of Benchmark: {cum_ret_b:.6f}\n")
    print(f"Standard Deviation of TOS: {std_daily_ret:.6f}")
    print(f"Standard Deviation of Benchmark: {std_daily_ret_b:.6f}\n")
    print(f"Average Daily Return of TOS: {avg_daily_ret:.6f}")
    print(f"Average Daily Return of Benchmark: {avg_daily_ret_b:.6f}\n")
    # print(f"Final Portfolio Value: {norm_portvals[-1]:.2f}")
    print(f"Final Portfolio Value of TOS: {portvals[-1]:.2f}")
    print(f"Final Portfolio Value of Benchmark: {benchmark_vals[-1]:.2f}")


    # STEP 5: Plot normalized portfolio vs benchmark
    plt.figure(figsize=(10, 6))
    plt.plot(norm_portvals.index, norm_portvals, label='Theoretically Optimal Strategy', color='red')
    plt.plot(benchmark_vals.index, benchmark_vals, label='Benchmark (1000 shares buy & hold)', color='purple')
    plt.title(f"Portfolio Value vs Benchmark for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("performance_comparison.png")
    plt.close()

# STEP 6: Author and study group info
def author():
    return "urafi3"

def study_group():
    return "urafi3"

def gtid():
    return 904074839

if __name__ == "__main__":
    testproject()


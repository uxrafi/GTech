"""
--------------------------------------------------------
# experiment1.py

This script runs Experiment 1 for ML4T project to compare the ManualStrategy 
and StrategyLearner against a Benchmark (buy-and-hold strategy) both 
in-sample (2008-2009) and out-of-sample (2010-2011).

- Initializes and trains the StrategyLearner on in-sample data.
- Runs ManualStrategy and StrategyLearner to generate trade decisions.
- Simulates portfolio values using generated trades and compares them 
  against a benchmark.
- Plots and saves in-sample and out-of-sample performance graphs.
- Prints performance metrics: Cumulative Return, Standard Deviation of 
  Daily Returns, Mean of Daily Returns.

# OUTPUT:
- experiment1_in_sample.png: Performance plot for in-sample period
- experiment1_out_sample.png: Performance plot for out-of-sample period
- Console output of metrics for all strategies in both periods.

# STEPS:
STEP 1: Import required libraries and modules
STEP 2: Define the run_experiment1() function
STEP 3: Define date ranges for in-sample and out-of-sample
STEP 4: Initialize ManualStrategy and StrategyLearner
STEP 5: Train StrategyLearner on in-sample data
STEP 6: Generate trades using both strategies for both periods
STEP 7: Define utility to compute portfolio values from trades
STEP 8: Compute in-sample and out-of-sample portfolio values
STEP 9: Plot and save in-sample and out-of-sample results
STEP 10: Calculate and print performance metrics
STEP 11: Execute if __main__
STEP 12: Return author information
--------------------------------------------------------
"""

# STEP 1: Import required libraries and modules
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from util import get_data

# STEP 2: Define the run_experiment1() function
def run_experiment1(symbol="JPM", sv=100000):
    """
    Main function to run Experiment 1 comparing strategies against benchmark
    """
    # STEP 3: Define date ranges
    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)
    out_sample_sd = dt.datetime(2010, 1, 1)
    out_sample_ed = dt.datetime(2011, 12, 31)
    
    # STEP 4: Initialize strategies
    print("Initializing strategies...")
    ms = ManualStrategy(verbose=True)  # Enable verbose mode for debugging
    sl = StrategyLearner()
    
    # STEP 5: Train StrategyLearner on in-sample data
    print("\nTraining StrategyLearner...")
    sl.add_evidence(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    
    # STEP 6: Generate trades for both strategies and periods
    print("\nGenerating trades...")
    ms_trades_is = ms.testPolicy(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    sl_trades_is = sl.testPolicy(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    ms_trades_oos = ms.testPolicy(symbol=symbol, sd=out_sample_sd, ed=out_sample_ed, sv=sv)
    sl_trades_oos = sl.testPolicy(symbol=symbol, sd=out_sample_sd, ed=out_sample_ed, sv=sv)

    # Debug prints to verify trades exist and show some entries
    print("\nManualStrategy In-Sample Trades (non-zero):")
    print(ms_trades_is[ms_trades_is[symbol] != 0])
    print("\nManualStrategy Out-of-Sample Trades (non-zero):")
    print(ms_trades_oos[ms_trades_oos[symbol] != 0])
    
    # STEP 7: Define improved function to compute portfolio values
    def compute_portvals(trades, symbol, sd, ed, sv):
        """Compute normalized portfolio values from trades"""
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates)
        prices = prices_all[[symbol]].ffill().bfill()  # Handle missing data
        
        # Special handling for benchmark (buy-and-hold)
        if len(trades) == 1:  # Benchmark case
            holdings = pd.DataFrame(1000, index=prices.index, columns=[symbol])
            trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
            trades.iloc[0] = 1000  # Initial purchase
        else:
            holdings = trades.cumsum()
        
        portvals = (holdings * prices) + (sv - (trades * prices).cumsum())
        return portvals / portvals.iloc[0]  # normalize to 1
    
    # STEP 8: Compute portfolio values with validation
    print("\nCalculating portfolio values...")
    ms_portvals_is = compute_portvals(ms_trades_is, symbol, in_sample_sd, in_sample_ed, sv)
    sl_portvals_is = compute_portvals(sl_trades_is, symbol, in_sample_sd, in_sample_ed, sv)
    benchmark_is = compute_portvals(pd.DataFrame(1000, index=[in_sample_sd], columns=[symbol]), 
                                  symbol, in_sample_sd, in_sample_ed, sv)
    
    ms_portvals_oos = compute_portvals(ms_trades_oos, symbol, out_sample_sd, out_sample_ed, sv)
    sl_portvals_oos = compute_portvals(sl_trades_oos, symbol, out_sample_sd, out_sample_ed, sv)
    benchmark_oos = compute_portvals(pd.DataFrame(1000, index=[out_sample_sd], columns=[symbol]), 
                                   symbol, out_sample_sd, out_sample_ed, sv)

    # STEP 9: Plot and save results with vertical lines for trades
    def plot_results(portvals_ms, portvals_sl, portvals_bm, trades_ms, filename, title):
        plt.figure(figsize=(14, 7))
        plt.plot(portvals_ms, label='Manual Strategy', color='red')
        plt.plot(portvals_sl, label='Strategy Learner', color='green')
        plt.plot(portvals_bm, label='Benchmark', color='purple')

        # Ensure trades_ms index is datetime
        trades_ms.index = pd.to_datetime(trades_ms.index)

        # Debug print to check trades shape and values
        print(f"\nPlotting vertical lines, number of trades: {len(trades_ms)}")
        print(trades_ms[trades_ms[symbol] != 0])

        # Draw vertical lines for LONG and SHORT entries
        for date, row in trades_ms.iterrows():
            trade_val = row[symbol]
            if trade_val > 0:
                plt.axvline(date, color='blue', linestyle='--', linewidth=1, alpha=0.7)
            elif trade_val < 0:
                plt.axvline(date, color='black', linestyle='--', linewidth=1, alpha=0.7)

        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Normalized Portfolio Value')
        plt.legend()
        plt.savefig(filename)
        plt.close()
    
    print("\nGenerating plots...")
    plot_results(ms_portvals_is, sl_portvals_is, benchmark_is, ms_trades_is, 
                 'experiment1_in_sample.png', 'In-Sample Performance Comparison')
    plot_results(ms_portvals_oos, sl_portvals_oos, benchmark_oos, ms_trades_oos,
                 'experiment1_out_sample.png', 'Out-of-Sample Performance Comparison')
    
    # STEP 10: Calculate and print metrics with validation
    def calculate_metrics(portvals):
        """Compute and return performance metrics"""
        if portvals.empty or len(portvals) < 2:
            return {'Cumulative Return': 0, 'Stdev Daily Returns': 0, 'Mean Daily Returns': 0}
        
        daily_returns = portvals.pct_change().dropna()
        if daily_returns.empty:
            return {'Cumulative Return': 0, 'Stdev Daily Returns': 0, 'Mean Daily Returns': 0}
            
        return {
            'Cumulative Return': portvals.iloc[-1].values[0] - 1,
            'Stdev Daily Returns': daily_returns.std().values[0],
            'Mean Daily Returns': daily_returns.mean().values[0]
        }
    
    print("\nIn-Sample Performance Metrics:")
    print("Manual Strategy:", calculate_metrics(ms_portvals_is))
    print("Strategy Learner:", calculate_metrics(sl_portvals_is))
    print("Benchmark:", calculate_metrics(benchmark_is))
    
    print("\nOut-of-Sample Performance Metrics:")
    print("Manual Strategy:", calculate_metrics(ms_portvals_oos))
    print("Strategy Learner:", calculate_metrics(sl_portvals_oos))
    print("Benchmark:", calculate_metrics(benchmark_oos))

# Add alias function for compatibility with testproject.py
def run():
    """Alias for run_experiment1 to maintain compatibility"""
    return run_experiment1()

# STEP 11: Execute experiment if run directly
if __name__ == "__main__":
    run_experiment1()

# STEP 12: Author information 
def author():
    return 'urafi3'

def study_group():
    return 'urafi3'

def gtid():
    return 904074839

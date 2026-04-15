"""
--------------------------------------------------------
# experiment2.py

This script runs Experiment 2 for the ML4T project to analyze the effect 
of market impact on the StrategyLearner's trading behavior and performance.

- Trains the StrategyLearner on the same in-sample data (2008-2009) 
  with different impact values.
- Measures the cumulative return, standard deviation, mean of daily returns, 
  and total trades for each impact setting.
- Plots normalized portfolio values for visual comparison across impact levels.
- Prints a summary table of performance metrics.

# OUTPUT:
- experiment2.png: A plot comparing portfolio values for different impact values.
- Console output summarizing performance metrics and number of trades for each impact level.

# STEPS:
STEP 1: Import required libraries and modules
STEP 2: Define the run_experiment2() function
STEP 3: Define in-sample date range
STEP 4: Define impact values to test
STEP 5: Loop over impact values to:
STEP 6: Plot normalized portfolio values for all impact levels
STEP 7: Print summary performance metrics in tabular form
STEP 8: Execute if __main__
STEP 9: Return author information
--------------------------------------------------------
"""


# STEP 1: Import required libraries and modules
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from StrategyLearner import StrategyLearner
from util import get_data

def run_experiment2(symbol="JPM", sv=100000):
    """
    Main function to analyze impact of market impact on strategy performance
    """
    # STEP 3: Define in-sample date range
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    
    # STEP 4: Define different impact values to test
    impact_values = [0.0, 0.005, 0.01]
    results = []
    
    # STEP 5: Test each impact value
    for impact in impact_values:
        # Initialize and train learner
        sl = StrategyLearner(impact=impact, commission=0.0)
        sl.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
        
        # Generate trades
        trades = sl.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
        
        # Compute portfolio values
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates)
        prices = prices_all[[symbol]]
        holdings = trades.cumsum()
        portvals = (holdings * prices) + (sv - (trades * prices).cumsum())
        portvals = portvals / portvals.iloc[0]  # Normalize
        
        # Calculate metrics
        daily_returns = portvals.pct_change().dropna()
        results.append({
            'Impact': impact,
            'Cumulative Return': portvals.iloc[-1].values[0] - 1,
            'Stdev Daily Returns': daily_returns.std().values[0],
            'Mean Daily Returns': daily_returns.mean().values[0],
            'Trades': trades.abs().sum().values[0],
            'Portfolio Values': portvals
        })
    
    # STEP 6: Plot results
    plt.figure(figsize=(14, 7))
    for result in results:
        plt.plot(result['Portfolio Values'], 
                label=f"Impact={result['Impact']}, Trades={result['Trades']}")
    plt.title('Strategy Performance vs. Impact')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.savefig('experiment2.png')
    plt.close()
    
    # STEP 7: Print results table
    print("\nExperiment 2 Results: Impact Analysis")
    print("="*80)
    print(f"{'Impact':<10}{'Cum Return':<15}{'Stdev Daily':<15}{'Mean Daily':<15}{'Trades':<10}")
    print("-"*80)
    for result in results:
        print(f"{result['Impact']:<10.4f}{result['Cumulative Return']:<15.4f}"
              f"{result['Stdev Daily Returns']:<15.6f}{result['Mean Daily Returns']:<15.6f}"
              f"{result['Trades']:<10}")
    print("="*80)

# Add alias function for compatibility with testproject.py
def run():
    """Alias for run_experiment2 to maintain compatibility"""
    return run_experiment2()

# STEP 8: Execute if run directly
if __name__ == "__main__":
    run_experiment2()

# STEP 9: Author information
def author():
    return 'urafi3'

def study_group():
    return 'urafi3'

def gtid():
    return 904074839
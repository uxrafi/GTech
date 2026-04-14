"""
Project 6 - marketsimcode.py
=============================

This file implements a market simulator that calculates the daily portfolio value
based on trades provided in a DataFrame. The simulator handles:
- Executing trades (buy/sell) on each date
- Adjusting for transaction cost and market impact
- Tracking cash and holdings over time
- Computing the final portfolio value series

Expected usage:
---------------
The file defines `compute_portvals()` which can be imported and used by other
scripts such as TheoreticallyOptimalStrategy.py or testproject.py.

This simulator supports a theoretical strategy scenario where transaction costs
and market impact may be zero.

Input:
------
- Trades DataFrame: Indexed by date, single column (symbol), with values representing
  +1000 for buy and -1000 for sell.
- Starting portfolio value (e.g., $100,000)
- Commission and impact (both float)

Output:
-------
- DataFrame with daily portfolio values

Author: urafi3
"""

# ========================================
# STEP 1: Import libraries and get price data
# STEP 2: Apply trades to calculate daily cash flow
# STEP 3: Accumulate holdings over time
# STEP 4: Compute portfolio value from cash and holdings
# STEP 5: Return result
# ========================================

import pandas as pd
import numpy as np
from util import get_data

def compute_portvals(
    trades,
    start_val=100000,
    commission=0.0,
    impact=0.0
):
    """
    Compute daily portfolio values based on trades and price data.

    Parameters:
        trades (pd.DataFrame): DataFrame with trades. Indexed by date, column = symbol.
        start_val (float): Starting cash value
        commission (float): Fixed commission cost per trade
        impact (float): Market impact cost as a percentage

    Returns:
        pd.DataFrame: Portfolio value for each day, indexed by date
    """

    # STEP 1: Extract symbol and price data
    symbol = trades.columns[0]  # Assume only one symbol (e.g., 'JPM')
    dates = trades.index
    prices = get_data([symbol], dates)  # Load price data
    prices = prices[[symbol]]           # Keep only the symbol column

    # STEP 2: Apply trades to calculate daily cash impact
    trades_df = trades.copy()
    trades_df['Cash'] = 0.0  # Track cash impact of trades

    for date in trades.index:
        shares = trades.loc[date, symbol]
        if shares == 0:
            continue  # No trade on this date

        price = prices.loc[date, symbol]
        
        # Calculate impact based on trade direction
        trade_impact = price * impact if shares > 0 else -price * impact
        adj_price = price + trade_impact

        # Calculate trade cost including commission
        cost = adj_price * shares + commission * np.sign(shares)

        # Deduct cost from cash
        trades_df.loc[date, 'Cash'] -= cost

    # STEP 3: Accumulate shares and cash over time
    holdings = trades_df.cumsum()
    holdings['Cash'] += start_val  # Add initial cash to first day's cash balance

    # STEP 4: Compute daily portfolio value
    portvals = (holdings[symbol] * prices[symbol]) + holdings['Cash']
    portvals = pd.DataFrame(portvals, columns=['Portfolio Value'])

    return portvals

# STEP 5: Author and study group info
def author():
    return "urafi3"

def study_group():
    return "urafi3"

def gtid():
    return 904074839

if __name__ == '__main__':
    print("This file is intended to be imported and used, not run directly.")

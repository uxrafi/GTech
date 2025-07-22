"""
TheoreticallyOptimalStrategy.py

This module implements a Theoretically Optimal Strategy (TOS)
for a given stock symbol over a specified time range.

Purpose:
The TOS algorithm assumes perfect knowledge of future prices
and attempts to maximize return by buying before price increases
and selling before price decreases. It is used to establish
an upper-bound benchmark for achievable portfolio performance.

This strategy:
- Only trades in increments of 1000 shares
- Maintains a maximum position of ±1000 shares
- May trade up to ±2000 shares in a day to reverse position
- Starts with $100,000 in cash and no shares
- Assumes no commission or market impact
- Closes all positions by the final trading day

This file defines:
- testPolicy(): returns a trades DataFrame following the TOS rules
- author(): returns GT username
- study_group(): returns GT usernames of collaborators (if any)

This code conforms to the Project 6 API requirements and can be
invoked by testproject.py as the entry point for evaluation.

"""


###############################################
# STEP 1: Import required libraries
# STEP 2: Define testPolicy() function
# STEP 3: Add author() and study_group() functions
# STEP 4: Optional test block to preview trades
###############################################


# STEP 1: Import required libraries
import pandas as pd         # for data manipulation
import numpy as np          # for numerical operations
import datetime as dt       # for date handling
from util import get_data   # utility function to fetch stock data


# STEP 2: Define the testPolicy function that returns a trades DataFrame
def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    """
    Compute a theoretically optimal trading policy for a given symbol.

    Parameters:
        symbol - Stock symbol to trade (e.g., 'JPM')
        sd     - Start date (datetime object)
        ed     - End date (datetime object)
        sv     - Starting portfolio value (default: 100000)

    Returns:
        trades - DataFrame of trades indexed by date, with single column = symbol
    """

    # Get price data from util.get_data() and forward/backward fill missing values
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates)[symbol]
    prices = prices.ffill().bfill()

    # Initialize trades DataFrame with same index as prices
    trades = pd.DataFrame(index=prices.index, data=0.0, columns=[symbol])

    position = 0  # Current holdings: can be -1000, 0, or +1000

    for i in range(len(prices) - 1):  # Look ahead one day
        date = prices.index[i]
        next_date = prices.index[i + 1]

        price_today = prices.loc[date]
        price_tomorrow = prices.loc[next_date]

        if price_tomorrow > price_today:
            # Expect price to rise → go long
            if position == 0:
                trades.loc[date, symbol] = 1000
                position = 1000
            elif position == -1000:
                trades.loc[date, symbol] = 2000
                position = 1000

        elif price_tomorrow < price_today:
            # Expect price to fall → go short
            if position == 0:
                trades.loc[date, symbol] = -1000
                position = -1000
            elif position == 1000:
                trades.loc[date, symbol] = -2000
                position = -1000
        else:
            # No price change → do nothing
            trades.loc[date, symbol] = 0.0

    # Close any open position on the last day
    if position != 0:
        trades.iloc[-1] = -position  # Ensure ending position is zero

    return trades


# STEP 3: Author and study group information
def author():
    return "urafi3"

def study_group():
    return "urafi3"

def gtid():
    return 904074839


# STEP 4: Optional test preview
if __name__ == "__main__":
    # Test the strategy for JPM during 2008–2009
    import datetime as dt
    df_trades = testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    print(df_trades.head(10))  # Show first 10 rows of trades

"""
--------------------------------------------------------
# ManualStrategy.py

This file implements a Manual Trading Strategy using technical indicators to guide buy, sell, and 
hold decisions.  This is part of the ML4T course final project, where we compare human-crafted strategies to machine 
learning strategies.


# WHAT THIS CLASS DOES:

- This file implementes a pre-defined API. It uses common technical indicators to evaluate stock price behavior:
  1. Simple Moving Average (SMA)
  2. Bollinger Bands (BB)
  3. Momentum
  4. Moving Average Convergence Divergence (MACD)

- Implements manual trading rules to generate trades based on the combined behavior of these indicators.
- Tracks and executes trades while ensuring the portfolio stays within allowed position limits (
  1000, 0, 1000 shares).
- Produces a **trades DataFrame** where:
  - +1000 = BUY 1000 shares
  - -1000 = SELL 1000 shares
  - 0 = HOLD (no action)
- Can be tested on any stock symbol and date range, but in the project we primarily use JPM.

# OUTPUT 

This class does not output or print anything to the console

# CODE FLOW:

# STEP 1: Import necessary libraries
# STEP 2: Define ManualStrategy class
# STEP 3: Load price data
# STEP 4: Calculate technical indicators
# STEP 5: Initialize trades DataFrame
# STEP 6: Track current position
# STEP 7: Apply manual trading rules
# STEP 8: Return the trades DataFrame
--------------------------------------------------------
"""

# STEP 1: Import necessary libraries
import numpy as np                  # for numerical operations
import pandas as pd                 # for data manipulation and analysis
import datetime as dt               # for handling dates
from util import get_data           # for loading stock data
from indicators import sma, momentum, bollinger_bands, macd  # for technical indicators


# STEP 2: Define ManualStrategy class
class ManualStrategy:
    # Initialize parameters
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """
        Manual strategy using technical indicators.
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    # The core strategy logic
    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
        """
        Generates trade signals using manually defined rules based on technical indicators.
        """
        # STEP 3: Load price data
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates)
        prices = prices_all[[symbol]]

        # STEP 4: Calculate technical indicators
        sma_values = sma(prices, window=20)
        bb_values, bb_upper, bb_lower = bollinger_bands(prices, window=20)
        momentum_values = momentum(prices, window=10)
        macd_values, macd_signal = macd(prices)

        # STEP 5: Initialize trades DataFrame
        trades = pd.DataFrame(0, index=prices.index, columns=[symbol])

        # STEP 6: Track current position
        current_position = 0  # 0 = out, 1000 = long, -1000 = short

        # STEP 7: Apply manual trading rules
        for i in range(1, len(prices)):
            price = prices.iloc[i][symbol]
            date = prices.index[i]
            bb_val = bb_values.iloc[i]
            mom_val = momentum_values.iloc[i]
            macd_diff = macd_values.iloc[i] - macd_signal.iloc[i]

            # STEP 7a: BUY signal
            if current_position == 0:
                if bb_val < -1.0 and mom_val > 0 and macd_diff > 0:
                    trades.loc[date, symbol] = 1000
                    current_position = 1000
            # STEP 7b: SELL signal
                elif bb_val > 1.0 and mom_val < 0 and macd_diff < 0:
                    trades.loc[date, symbol] = -1000
                    current_position = -1000
            # STEP 7c: Exit Long position
            elif current_position == 1000:
                if bb_val > 0.5 or mom_val < 0 or macd_diff < 0:
                    trades.loc[date, symbol] = -1000
                    current_position = 0
            # STEP 7d: Exit Short position
            elif current_position == -1000:
                if bb_val < -0.5 or mom_val > 0 or macd_diff > 0:
                    trades.loc[date, symbol] = 1000
                    current_position = 0

        # STEP 8: Return the trades DataFrame
        return trades

    # Author function
    def author():
        return 'urafi3'

    def study_group():
        return 'urafi3' 
    
    def gtid():
        return 904074839
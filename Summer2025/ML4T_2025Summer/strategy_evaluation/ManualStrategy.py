'''
--------------------------------------------------------
# ManualStrategy.py

This file implements a Manual Trading Strategy using technical indicators to guide buy, sell, and 
hold decisions.  

- Implements manual trading rules to generate trades based on the combined behavior of these indicators.
- Tracks and executes trades while ensuring the portfolio stays within allowed position limits (
  1000, 0, 1000 shares).
- Produces a trades DataFrame where:
  - +1000 = BUY 1000 shares
  - -1000 = SELL 1000 shares
  - 0 = HOLD (no action)

# INDICATORS USED:
It uses common technical indicators (from indicatory.py) to evaluate stock price behavior:
  1. Bollinger Bands (BB)
  2. Momentum
  3. Moving Average Convergence Divergence (MACD)

# OUTPUT 
This class does not output or print anything to the console unless verbose=True

# STEPS:
# STEP 1: Import necessary libraries
# STEP 2: Define ManualStrategy class
# STEP 3: Load price data
# STEP 4: Calculate technical indicators
# STEP 5: Initialize trades DataFrame
# STEP 6: Track current position
# STEP 7: Apply manual trading rules
# STEP 8: Return the trades DataFrame
--------------------------------------------------------
'''

# STEP 1: Import necessary libraries
import numpy as np  # for numerical operations
import pandas as pd # for data manipulation
import datetime as dt # for date handling
from util import get_data   # for loading stock data
from indicators import momentum, bollinger_bands, macd # for technical indicators

# STEP 2: Define ManualStrategy class
class ManualStrategy:
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def add_evidence(self, symbol='IBM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), sv=100000):
        """
        For consistency with Strategy Learner; no training needed for manual strategy
        """
        pass

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
        """
        Tests the manual trading policy on data between sd and ed

        Parameters:
            symbol (str): Stock symbol to trade
            sd (datetime): Start date
            ed (datetime): End date
            sv (int): Starting portfolio value (not used here)

        Returns:
            trades (pd.DataFrame): Single column dataframe of trades (+1000, -1000, 0, +2000, -2000)
        """

        # STEP 3: Load price data with error handling for missing data files
        dates = pd.date_range(sd, ed)
        try:
            prices_all = get_data([symbol], dates)
        except FileNotFoundError:
            raise ValueError(f"No price data found for {symbol} from {sd} to {ed}")
        
        prices = prices_all[[symbol]].ffill().bfill()
        
        if prices.empty:
            raise ValueError(f"No price data found for {symbol} from {sd} to {ed}")

        # STEP 4: Calculate technical indicators
        bb_val, _, _, _ = bollinger_bands(prices, window=20)
        momentum_values = momentum(prices, window=10)
        macd_values = macd(prices)

        # STEP 5: Initialize trades DataFrame and current position
        trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
        current_position = 0

        # STEP 6 & 7: Apply manual trading rules starting after indicator warmup period (20 days)
        for i in range(20, len(prices)):
            date = prices.index[i]
            try:
                bb = float(bb_val.iloc[i])
                mom = float(momentum_values.iloc[i])
                macd_val = float(macd_values.iloc[i])
            except (ValueError, IndexError):
                continue  # Skip if data invalid for any reason

            # Define buy/sell signals with relaxed thresholds for sensitivity
            buy_signal = (bb < -0.5) or (mom > 0.005 and macd_val > 0.005)
            sell_signal = (bb > 0.5) or (mom < -0.005 and macd_val < -0.005)

            # STEP 7: Position management and trade generation
            if current_position == 0:  # Neutral position
                if buy_signal and not sell_signal:
                    trades.loc[date, symbol] = 1000
                    current_position = 1000
                elif sell_signal and not buy_signal:
                    trades.loc[date, symbol] = -1000
                    current_position = -1000
            elif current_position > 0:  # Long position
                if sell_signal:
                    trades.loc[date, symbol] = -2000  # Close long, open short
                    current_position = -1000
                elif bb > 0.8 or mom < -0.01 or macd_val < -0.01:
                    trades.loc[date, symbol] = -1000  # Close long
                    current_position = 0
            else:  # Short position
                if buy_signal:
                    trades.loc[date, symbol] = 2000  # Close short, open long
                    current_position = 1000
                elif bb < -0.8 or mom > 0.01 or macd_val > 0.01:
                    trades.loc[date, symbol] = 1000  # Close short
                    current_position = 0

            # Output for debugging trades
            if self.verbose and trades.loc[date, symbol] != 0:
                action = "BUY" if trades.loc[date, symbol] > 0 else "SELL"
                print(f"{date.date()}: {action} {abs(trades.loc[date, symbol])} shares "
                      f"(BB={bb:.2f}, Mom={mom:.4f}, MACD={macd_val:.4f})")

        # STEP 8: Close any open position on the last day
        if current_position != 0:
            trades.iloc[-1, 0] = -current_position
            if self.verbose:
                print(f"Final closing trade on {trades.index[-1].date()}: {trades.iloc[-1, 0]} shares")

        # Summary output if verbose
        if self.verbose:
            trade_counts = trades[trades[symbol] != 0]
            print(f"\nTrade Summary ({sd.date()} to {ed.date()}):")
            print(f"Total trades: {len(trade_counts)}")
            print(f"Buy signals: {(trades[symbol] > 0).sum()}")
            print(f"Sell signals: {(trades[symbol] < 0).sum()}")
            print("Sample trades:")
            print(trades[trades[symbol] != 0].head(10))

        return trades

    # author information 
    def author():
        return 'urafi3'

    def study_group():
        return 'urafi3'

    def gtid():
        return 904074839

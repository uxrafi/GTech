"""
--------------------------------------------------------
# StrategyLearner.py

This module defines the StrategyLearner class, which implements an "automated trading strategy 
using machine learning." It is part of the ML4T project for building and comparing manual vs. 
learned strategies.

- Trains a trading policy using a machine learning ensemble approach (Bagging + Random Tree Learner)
- Uses the trained model to predict and generate trades (BUY/SELL/HOLD) for a given stock on specific dates

# INDICATORS USED:
- Uses the same three indicators (from indicators.py) as the Manual Strategy for consistency and fair comparison:
    1. Bollinger Bands (BB)
    2. Momentum
    3. MACD

# LEARNERS USED:
- BagLearner: Uses an ensemble of RTLearners to reduce overfitting and capture complex data patterns.
- RTLearner: Decision tree-based model that handles non-linear relationships in features.

# OUTPUT:
- Produces a trades DataFrame indicating buy/sell/hold decisions (+1000/-1000/0)
- This output is later passed to marketsimcode.py to simulate portfolio performance.

# STEPS:
STEP 1: Import required libraries
STEP 2: Define the StrategyLearner class and initialize parameters
STEP 3: Train the learner using add_evidence()
STEP 4: Generate trades using testPolicy()
STEP 5: Author functions to provide metadata about the author and study groups

--------------------------------------------------------
"""

# STEP 1: Import required libraries
import numpy as np                      # For numerical operations on arrays
import pandas as pd                     # For data manipulation with DataFrames
import datetime as dt                   # For handling dates and date ranges
from util import get_data               # Custom utility to retrieve stock price data
from indicators import momentum, bollinger_bands, macd  # Only 3 indicators used
from BagLearner import BagLearner       # Ensemble learner that uses multiple learners (bagging)
from RTLearner import RTLearner         # Random Tree learner for decision-making


# STEP 2: Define the StrategyLearner class and initialize parameters
class StrategyLearner:
    def __init__(self, verbose=False, impact=0.005, commission=9.95, num_bags=20, leaf_size=5):
        """
        Initialize the StrategyLearner.

        Parameters:
        verbose (bool): If True, print debug information.
        impact (float): Market impact cost per trade.
        commission (float): Fixed commission cost per trade.
        num_bags (int): Number of trees in the BagLearner ensemble.
        leaf_size (int): Leaf size for RTLearners (>=5 for classification stability).
        """
        # Store parameters
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.num_bags = num_bags
        self.leaf_size = max(5, leaf_size)  # Ensure leaf size >= 5 per project guidelines
        self.learner = None  # Will hold trained BagLearner ensemble

        # Indicator parameters - must match ManualStrategy
        self.momentum_window = 10
        self.bb_window = 20
        self.macd_short = 12
        self.macd_long = 26
        self.bins = 10  # Number of bins to discretize continuous future returns into classes


    # STEP 3: Train the learner using in-sample data
    def add_evidence(self, symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
        """
        Train the StrategyLearner on the in-sample period.

        Parameters:
        symbol (str): Stock symbol to train on.
        sd (datetime): Start date of training.
        ed (datetime): End date of training.
        sv (int): Starting portfolio value (not used in this method).
        """
        # 1. Fetch stock prices over the training date range
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates)
        prices = prices_all[[symbol]]

        # 2. Calculate the three indicators matching ManualStrategy
        momentum_val = momentum(prices, window=self.momentum_window)
        bb_val, _, _, _ = bollinger_bands(prices, window=self.bb_window)
        macd_val = macd(prices, span_short=self.macd_short, span_long=self.macd_long)

        # 3. Combine the indicator values into a feature matrix
        features = pd.concat([momentum_val, bb_val, macd_val], axis=1)
        features.columns = ['Momentum', 'BB', 'MACD']
        features = features.dropna()  # Drop rows with NaN (due to indicator warmup periods)

        # 4. Compute future returns over next 5 trading days as prediction targets
        future_returns = prices.shift(-5) / prices - 1
        future_returns = future_returns.loc[features.index]  # Align with features by index
        # 5. Adjust future returns for market impact (penalize returns by impact cost)
        future_returns[symbol] = future_returns[symbol] - self.impact

        # 6. Discretize the continuous future returns into 'bins' classes for classification
        returns_binned = pd.qcut(future_returns[symbol], self.bins, labels=False, duplicates='drop')

        # 7. Initialize BagLearner using RTLearner as the base learner
        self.learner = BagLearner(
            learner=RTLearner,
            kwargs={"leaf_size": self.leaf_size},
            bags=self.num_bags,
            boost=False,
            verbose=self.verbose
        )

        # 8. Train the BagLearner ensemble on the features and discretized returns
        self.learner.add_evidence(features.values, returns_binned.values)

        if self.verbose:
            print(f"StrategyLearner training complete with {self.num_bags} RTLearners.")


    # STEP 4: Generate trades using the trained model on the given test period
    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=100000):
        """
        Use the trained model to generate trades in the testing period.

        Parameters:
        symbol (str): Stock symbol to test on.
        sd (datetime): Start date of testing.
        ed (datetime): End date of testing.
        sv (int): Starting portfolio value (not used here).

        Returns:
        pd.DataFrame: Trades DataFrame indexed by date with column symbol.
                      Values are +1000 (buy), -1000 (sell), or 0 (hold).
        """
        # Raise error if learner is not yet trained
        if self.learner is None:
            raise Exception("StrategyLearner must be trained before calling testPolicy().")

        # 1. Fetch stock prices for testing period
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates)
        prices = prices_all[[symbol]]

        # 2. Recalculate the same three indicators on the test prices
        momentum_val = momentum(prices, window=self.momentum_window)
        bb_val, _, _, _ = bollinger_bands(prices, window=self.bb_window)
        macd_val = macd(prices, span_short=self.macd_short, span_long=self.macd_long)

        # 3. Combine into feature matrix and drop NaNs due to indicator calculation
        features = pd.concat([momentum_val, bb_val, macd_val], axis=1)
        features.columns = ['Momentum', 'BB', 'MACD']
        features = features.dropna()

        # 4. Initialize trades DataFrame with zeros
        trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
        position = 0  # Current holdings (long=1000, short=-1000, flat=0)

        # 5. Predict classification bins for each day using the trained learner
        pred_bins = self.learner.query(features.values)

        # 6. Convert predicted bins into discrete trading positions:
        # Use top 20% bins to signal long, bottom 20% to signal short, others neutral
        top_threshold = self.bins * 0.8
        bottom_threshold = self.bins * 0.2

        # 7. Iterate over predictions to decide trades needed to reach target position
        for i in range(len(pred_bins)):
            predicted_bin = pred_bins[i]

            # --- FIX: ensure predicted_bin is numeric (convert to float in case of string type) ---
            predicted_bin = float(predicted_bin)

            if predicted_bin >= top_threshold:
                target_position = 1000  # Go long
            elif predicted_bin <= bottom_threshold:
                target_position = -1000  # Go short
            else:
                target_position = 0  # Neutral

            shares_to_trade = target_position - position

            # Place trade only if position needs to change
            if shares_to_trade != 0:
                trades.loc[features.index[i], symbol] = shares_to_trade
                position = target_position  # Update current position

        return trades


    # STEP 5: Author information 
    def author():
        """
        Return Georgia Tech user ID for grading.
        """
        return 'urafi3'

    def study_group():
        """
        Return study group members as comma-separated string.
        """
        return 'urafi3'

    def gtid():
        """
        Return GT ID number (optional).
        """
        return 904074839

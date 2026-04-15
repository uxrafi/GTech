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
STEP 5: Utility functions including author() and study_group()

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
        # Parameters
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.num_bags = num_bags
        self.leaf_size = max(5, leaf_size)  # Ensure a minimum leaf size for stability
        self.learner = None  # This will be the trained BagLearner model

        # Indicator parameters - same as ManualStrategy
        self.momentum_window = 10
        self.bb_window = 20
        self.macd_short = 12
        self.macd_long = 26
        self.bins = 10  # Number of bins to discretize future returns into classes


    # STEP 3: Train the learner using in-sample data
    def add_evidence(self, symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
        # Get stock price data
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates)
        prices = prices_all[[symbol]]

        # Compute only the three indicators
        momentum_val = momentum(prices, window=self.momentum_window)
        bb_val, _, _, _ = bollinger_bands(prices, window=self.bb_window)
        macd_val = macd(prices, span_short=self.macd_short, span_long=self.macd_long)

        # Combine indicators into a feature matrix
        features = pd.concat([momentum_val, bb_val, macd_val], axis=1)
        features.columns = ['Momentum', 'BB', 'MACD']
        features = features.dropna()  # Remove any NaN entries caused by indicator calculations

        # Calculate future 5-day returns as the prediction target
        future_returns = prices.shift(-5) / prices - 1
        future_returns = future_returns.loc[features.index]  # Align with features
        future_returns[symbol] = future_returns[symbol] - self.impact  # Account for market impact

        # Discretize continuous returns into bins for classification
        returns_binned = pd.qcut(future_returns[symbol], self.bins, labels=False, duplicates='drop')

        # Initialize BagLearner with RTLearner as the base learner
        self.learner = BagLearner(
            learner=RTLearner,
            kwargs={"leaf_size": self.leaf_size},
            bags=self.num_bags,
            boost=False,
            verbose=self.verbose
        )

        # Train the model using features (X) and binned returns (y)
        self.learner.add_evidence(features.values, returns_binned.values)


    # STEP 4: Generate trades using the trained model on the given test period
    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=100000):
        # Get stock price data
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates)
        prices = prices_all[[symbol]]

        # Recalculate indicators for the test period
        momentum_val = momentum(prices, window=self.momentum_window)
        bb_val, _, _, _ = bollinger_bands(prices, window=self.bb_window)
        macd_val = macd(prices, span_short=self.macd_short, span_long=self.macd_long)

        # Combine into features matrix
        features = pd.concat([momentum_val, bb_val, macd_val], axis=1)
        features.columns = ['Momentum', 'BB', 'MACD']
        features = features.dropna()

        # Prepare to store trades
        trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
        position = 0  # Current position held (long: 1000, short: -1000, neutral: 0)

        # Query the trained model for predictions
        testX = features.values
        predY = self.learner.query(testX)

        # Convert predicted bins into trade decisions
        for i in range(len(predY)):
            predicted_bin = predY[i]
            if predicted_bin >= self.bins * 0.8:
                target_position = 1000  # Strong positive prediction: Go long
            elif predicted_bin <= self.bins * 0.2:
                target_position = -1000  # Strong negative prediction: Go short
            else:
                target_position = 0  # Neutral prediction: No position

            # Determine number of shares to trade to achieve the target position
            shares_to_trade = target_position - position
            if shares_to_trade != 0:
                trades.loc[features.index[i], symbol] = shares_to_trade
                position = target_position  # Update current position

        return trades


    # STEP 5: Utility functions to provide metadata
    def author():
        return 'urafi3'

    def study_group():
        return 'urafi3'

    def gtid():
        return 904074839

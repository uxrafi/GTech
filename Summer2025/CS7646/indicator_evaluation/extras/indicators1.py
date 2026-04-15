"""
Project 6 - indicators.py
==========================

This file is responsible for calculating and visualizing five technical indicators
used to analyze stock price trends and generate insights for developing trading strategies.
Each function in this file returns a single real-valued vector (as required).

The file can be run standalone to generate plots for each indicator based on the JPM stock
from January 1, 2008 to December 31, 2009. The generated plots are saved as PNG files .

Indicators Implemented:
-----------------------
1. Simple Moving Average (SMA) - smooths price over a window to reveal trend.
2. Exponential Moving Average (EMA) - similar to SMA, but emphasizes recent prices.
3. Momentum - measures the rate of price change over time.
4. Bollinger Bands Value (BB) - indicates how far price is from the moving average, scaled by volatility.
5. MACD - measures the difference between two EMAs to identify trend changes.

Each indicator is calculated and plotted with proper labels and saved to file (one
for each indicator plus one for portfolio value comparison).

These indicators will also be reused in Project 8 as input features for strategy learning.

Author: urafi3
"""

###################################
# STEP 1: Import necessary libraries and functions
# STEP 2: Define indicator functions
# STEP 3: Run function to calculate and plot indicators
# STEP 4: Plot each indicator
# STEP 5: Author and study group info
###################################

# STEP 1: Import necessary libraries and functions
import pandas as pd             # for data manipulation   
import numpy as np              # for numerical operations
import matplotlib.pyplot as plt # for plotting
from util import get_data       # utility function to fetch stock data

# STEP 2: Define indicator functions
def sma(prices, window=14):
    # Calculate Simple Moving Average using rolling mean
    # Create a rolling window of the specified size (default 14)
    # Then compute the mean within each window to get the Simple Moving Average
    return prices.rolling(window=window).mean()  # pandas' rolling() function

def ema(prices, span=14):
    # Calculate Exponential Moving Average with specified span
    return prices.ewm(span=span, adjust=False).mean()

def momentum(prices, window=10):
    # Calculate Momentum as (P_t / P_(t-N)) - 1
    return (prices / prices.shift(window)) - 1

def bollinger_bands(prices, window=20):
    # Calculate Bollinger Band value: (Price - SMA) / (2 * Std Dev)
    sma_val = sma(prices, window)
    std = prices.rolling(window=window).std()
    bb_val = (prices - sma_val) / (2 * std)
    return bb_val

def macd(prices, span_short=12, span_long=26):
    # Calculate MACD as difference between short and long EMA
    ema_short = ema(prices, span=span_short)
    ema_long = ema(prices, span=span_long)
    macd_val = ema_short - ema_long
    return macd_val

# STEP 3: Run function to calculate and plot indicators
def run():
    symbol = "JPM"
    sd = pd.to_datetime("2008-01-01")
    ed = pd.to_datetime("2009-12-31")
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates)[symbol]  # Load adjusted close prices

    # Calculate all indicators using JPM price data
    indicators = {
        "Simple Moving Average": sma(prices, window=20),
        "Exponential Moving Average": ema(prices, span=20),
        "Momentum": momentum(prices, window=10),
        "Bollinger Bands": bollinger_bands(prices, window=20),
        "MACD": macd(prices),
    }

    # STEP 4: Plot each indicator
    for name, indicator in indicators.items():
        plt.figure(figsize=(10, 6))

        if name in ["Simple Moving Average", "Exponential Moving Average"]:
            # Normalize both price and indicator for visual comparison
            plt.plot(prices / prices.iloc[0], label="Normalized Price")
            plt.plot(indicator / indicator.iloc[0], label=f"Normalized {name}")
        else:
            # Plot the indicator directly
            plt.plot(indicator, label=name)
            plt.axhline(y=0, color='gray', linestyle='--')  # Horizontal reference line

        plt.title(f"{name} Indicator for {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to file with lowercase, underscore filename
        filename = f"{name.replace(' ', '_').lower()}.png"
        plt.savefig(filename)
        plt.close()

# STEP 5: Author and study group info
def author():
    return "urafi3"

def study_group():
    return "urafi3"

def gtid():
    return 904074839

if __name__ == "__main__":
    run()

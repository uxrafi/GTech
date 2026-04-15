"""
Project 6 - indicators.py
==========================

This file is responsible for calculating and visualizing five technical indicators
used to analyze stock price trends and generate insights for developing trading strategies.
Each function in this file returns a single real-valued vector (as required).

The file can be run standalone to generate plots for each indicator based on the JPM stock
from January 1, 2008 to December 31, 2009. The generated plots are saved as PNG files.

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

# STEP 1: Import necessary libraries and functions
import pandas as pd             # for data manipulation
import numpy as np              # for numerical operations
import matplotlib.pyplot as plt # for plotting
from util import get_data       # utility function to fetch stock data

# STEP 2: Define indicator functions
def sma(prices, window=14):
    # Calculate Simple Moving Average using rolling mean
    return prices.rolling(window=window).mean()

def ema(prices, span=14):
    # Calculate Exponential Moving Average with specified span
    return prices.ewm(span=span, adjust=False).mean()
    

def momentum(prices, window=10):
    # Calculate Momentum as (P_t / P_(t-N)) - 1
    return (prices / prices.shift(window)) - 1

def bollinger_bands(prices, window=20):
    """
    Calculate Bollinger Band value and components.
    Returns:
        bb_val: (price - SMA) / (2 * std)
        sma_val: simple moving average
        upper_band: SMA + 2 * std
        lower_band: SMA - 2 * std
    """
    sma_val = sma(prices, window)
    std = prices.rolling(window=window).std()
    upper_band = sma_val + 2 * std
    lower_band = sma_val - 2 * std
    bb_val = (prices - sma_val) / (2 * std)
    return bb_val, sma_val, upper_band, lower_band

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

    # Calculate indicators
    sma_val = sma(prices, window=20)
    # ema_val = ema(prices, span=20)
    ema_val = ema(prices, span=10)  # Lower span makes EMA more responsive
    momentum_val = momentum(prices, window=10)
    bb_val, bb_sma, bb_upper, bb_lower = bollinger_bands(prices, window=20)
    macd_val = macd(prices)

# Plot SMA
    plt.figure(figsize=(10, 6))
    plt.plot(prices / prices.iloc[0], label="Normalized Price")

    # Avoid division by NaN
    sma_first_valid = sma_val[sma_val.first_valid_index()]
    plt.plot(sma_val / sma_first_valid, label="Normalized SMA")

    plt.title("Simple Moving Average (SMA) for JPM")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("simple_moving_average.png")
    plt.close()

    # Plot EMA
    plt.figure(figsize=(10, 6))
    plt.plot(prices / prices.iloc[0], label="Normalized Price")
    plt.plot(ema_val / ema_val.iloc[0], label="Normalized EMA")
    plt.title("Exponential Moving Average (EMA) for JPM")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("exponential_moving_average.png")
    plt.close()

    # Plot Momentum
    plt.figure(figsize=(10, 6))
    plt.plot(momentum_val, label="Momentum")
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title("Momentum Indicator for JPM")
    plt.xlabel("Date")
    plt.ylabel("Momentum")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("momentum.png")
    plt.close()

    # Plot Bollinger Bands with Price, SMA, Upper & Lower Bands
    plt.figure(figsize=(10, 6))
    plt.plot(prices, label="Price")
    plt.plot(bb_sma, label="SMA", linestyle="--")
    plt.plot(bb_upper, label="Upper Band", linestyle=":")
    plt.plot(bb_lower, label="Lower Band", linestyle=":")
    plt.title("Bollinger Bands for JPM")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bollinger_bands.png")
    plt.close()

    # Plot MACD
    plt.figure(figsize=(10, 6))
    plt.plot(macd_val, label="MACD")
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title("MACD Indicator for JPM")
    plt.xlabel("Date")
    plt.ylabel("MACD Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("macd.png")
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

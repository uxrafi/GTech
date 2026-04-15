"""Analyze a portfolio.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Copyright 2017, Georgia Tech Research Corporation  		  	   		 	 	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332-0415  		  	   		 	 	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  	

# STEPS
# 1. Import necessary libraries
# 2. Define the assess_portfolio function
# 3. Read in stock data for the specified symbols and date range
# 4. Compute daily portfolio values based on allocations and stock prices
# 5. Calculate performance metrics: cumulative return, average daily return, standard deviation of daily returns, Sharpe ratio
# 6. Generate a plot comparing portfolio performance with SPY   
# 7. Return the computed metrics and end value of the portfolio


# STEP 1: Import necessary libraries		  	   		 	 	 			  		 			 	 	 		 		 	
import datetime as dt                 # Provides date and time manipulation tools
import numpy as np                    # Supports numerical operations and array handling
import pandas as pd                   # Enables data manipulation and analysis with DataFrames
from util import get_data, plot_data  # Imports custom functions to fetch and plot stock data from util.py

  		  	   		 	 	 			  		 			 	 	 		 		 	
# STEP 2: Define the assess_portfolio function  		  	   		 	 	 			  		 			 	 	 		 		 	
# This is the function that will be tested by the autograder  

"""
The assess_portfolio function evaluates the performance of an investment portfolio over a given date 
range using specified stock allocations. 

- It calculates financial metrics that describe how well the portfolio performed, such as:
- Cumulative return
- Average daily return
- Standard deviation of daily returns (volatility)
- Sharpe ratio
- Portfolio value over time
"""

def assess_portfolio(
    sd=dt.datetime(2008, 1, 1),      # Start date of the portfolio
    ed=dt.datetime(2009, 1, 1),      # End date of the portfolio
    syms=["GOOG", "AAPL", "GLD", "XOM"],  # List of stock symbols
    allocs=[0.1, 0.2, 0.3, 0.4],     # Allocation to each stock (must sum to 1.0)
    sv=1000000,                      # Starting value of the portfolio
    rfr=0.0,                         # Risk-free rate (used to compute Sharpe ratio)
    sf=252.0,                        # Sampling frequency (typically 252 trading days/year)
    gen_plot=False,                  # If True, generates a plot of portfolio vs. SPY
):
		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    Estimate a set of test points given the model we built.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type sd: datetime  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type ed: datetime  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param syms: A list of 2 or more symbols that make up the portfolio (note that your code should support any symbol in the data directory)  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type syms: list  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param allocs:  A list of 2 or more allocations to the stocks, must sum to 1.0  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type allocs: list  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param sv: The starting value of the portfolio  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type sv: int  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param rfr: The risk free return per sample period that does not change for the entire date range (a single number, not an array)  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type rfr: float  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param sf: Sampling frequency per year  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type sf: float  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	 	 			  		 			 	 	 		 		 	
        code with gen_plot = False.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type gen_plot: bool  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: A tuple containing the cumulative return, average daily returns,  		  	   		 	 	 			  		 			 	 	 		 		 	
        standard deviation of daily returns, Sharpe ratio and end value  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: tuple  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	

# STEP 3. Read in stock data for the specified symbols and date range	  	   		 	 	 			  		 			 	 	 		 		 	 		  	   		 	 	 			  		 			 	 	 		 		 	
    dates = pd.date_range(sd, ed)  	# # Create a range of dates using pandas from start date (sd) to end date (ed)	  	   		 	 	 			  		 			 	 	 		 		 	
    prices_all = get_data(syms, dates)  # Fetch stock prices for symbols and dates; adds SPY automatically if not in syms  		  	   		 	 	 			  		 			 	 	 		 		 	
    prices = prices_all[syms]  # only portfolio symbols  		  	   		 	 	 			  		 			 	 	 		 		 	
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
   # Get daily portfolio value  		  	   		 	 	 			  		 			 	 	 		 		 	
   # port_val = prices_SPY  # add code here to compute daily portfolio values 


# STEP 4: Compute daily portfolio values based on allocations and stock prices.
    """
    Call the function to compute the daily portfolio value by applying allocations to stock prices and scaling by 
    starting value

    - prices come from the code above, which reads in the adjusted closing prices for the specified symbols
    - allocs is passed directly as a parameter from the calling method to assess_portfolio()
    - sv is passed directly as a parameter from the calling method to assess_portfolio()
    """
    port_val = compute_weighted_portfolio_value(prices, allocs, sv)
 		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # STEP 5. Calculate performance metrics: cumulative return, average daily return, standard deviation of daily returns, Sharpe ratio 		  	   		 	 	 			  		 			 	 	 		 		 	
    """
    cr, adr, sddr, sr = [  		  	   		 	 	 			  		 			 	 	 		 		 	
        0.25,  		  	   		 	 	 			  		 			 	 	 		 		 	
        0.001,  		  	   		 	 	 			  		 			 	 	 		 		 	
        0.0005,  		  	   		 	 	 			  		 			 	 	 		 		 	
        2.1,  		  	   		 	 	 			  		 			 	 	 		 		 	
    ]  # add code here to compute stats 

    """	  	   		 	 	 			  		 			 	 	 		 		 	
    
    cr, adr, sddr, sr = compute_performance_metrics(port_val, rfr, sf)


# STEP 6. Generate a plot comparing portfolio performance with SPY  
    if gen_plot:
        # Normalize values to start at 1.0
        norm_portfolio = port_val / port_val.iloc[0]
        norm_SPY = prices_SPY / prices_SPY.iloc[0]

        # Combine normalized values
        df_temp = pd.concat([norm_portfolio, norm_SPY], axis=1)
        df_temp.columns = ['Portfolio', 'SPY']

        # Plot using util's plot_data function
        plot_data(df_temp, title="Portfolio Value vs SPY", ylabel="Normalized Price")

    # Add code here to properly compute end value
    # ev = sv

# STEP 7. Calculate performance metrics: cumulative return, average daily return, standard deviation of daily returns, Sharpe ratio  
    ev = port_val.iloc[-1]  # ev is the last value in the portfolio series

    return cr, adr, sddr, sr, ev	  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	

def compute_weighted_portfolio_value(prices_df, allocation_vector, initial_investment):
    
    """
    Compute the daily portfolio value given prices and allocations.

    :param prices_df: DataFrame of daily adjusted closing prices for the portfolio stocks (dates as index, symbols as columns)
    :param allocation_vector: List or array of allocation percentages per stock (should sum to 1)
    :param initial_investment: Initial amount invested in the portfolio
    :return: Series representing the daily total value of the portfolio
    """
    # Normalize prices relative to the first day to get growth factors
    normalized_prices = prices_df / prices_df.iloc[0]

    # Multiply normalized prices by allocations to get weighted growth per stock
    weighted_growth = normalized_prices.multiply(allocation_vector, axis=1)

    # Sum across stocks to get total portfolio growth factor for each day
    portfolio_growth = weighted_growth.sum(axis=1)

    # Multiply by initial investment to get portfolio value series
    portfolio_value_series = portfolio_growth * initial_investment

    return portfolio_value_series


def compute_performance_metrics(portfolio_series, risk_free=0.0, periods_per_year=252.0):
    
    """
    Compute key performance metrics for a portfolio's value series:
    total return, mean daily return, daily return volatility, and risk-adjusted return.
    
    :param portfolio_series: pandas Series of portfolio values indexed by date
    :param risk_free: risk-free interest rate per period (default zero)
    :param periods_per_year: number of trading periods in a year (default 252)
    :return: tuple of (total_return, mean_return, volatility, risk_adjusted_return)
    """
    # Calculate daily percentage returns, skipping first NaN
    returns_daily = portfolio_series.pct_change().iloc[1:]
    
    # Total return over the period
    total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
    
    # Average daily return
    mean_return = returns_daily.mean()
    
    # Standard deviation of daily returns (volatility)
    volatility = returns_daily.std()
    
    # Calculate risk-adjusted return (Sharpe ratio)
    risk_adjusted_return = ((mean_return - risk_free) / volatility) * np.sqrt(periods_per_year)
    
    return total_return, mean_return, volatility, risk_adjusted_return


def test_code():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    Performs a test of your code and prints the results  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    # This code WILL NOT be tested by the auto grader  		  	   		 	 	 			  		 			 	 	 		 		 	
    # It is only here to help you set up and test your code  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Define input parameters  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Note that ALL of these values will be set to different values by  		  	   		 	 	 			  		 			 	 	 		 		 	
    # the autograder!  		  	   		 	 	 			  		 			 	 	 		 		 	
    start_date = dt.datetime(2009, 1, 1)  		  	   		 	 	 			  		 			 	 	 		 		 	
    end_date = dt.datetime(2010, 1, 1)  		  	   		 	 	 			  		 			 	 	 		 		 	
    symbols = ["GOOG", "AAPL", "GLD", "XOM"]  		  	   		 	 	 			  		 			 	 	 		 		 	
    allocations = [0.2, 0.3, 0.4, 0.1]  		  	   		 	 	 			  		 			 	 	 		 		 	
    start_val = 1000000  		  	   		 	 	 			  		 			 	 	 		 		 	
    risk_free_rate = 0.0  		  	   		 	 	 			  		 			 	 	 		 		 	
    sample_freq = 252  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Assess the portfolio  		  	   		 	 	 			  		 			 	 	 		 		 	
    cr, adr, sddr, sr, ev = assess_portfolio(  		  	   		 	 	 			  		 			 	 	 		 		 	
        sd=start_date,  		  	   		 	 	 			  		 			 	 	 		 		 	
        ed=end_date,  		  	   		 	 	 			  		 			 	 	 		 		 	
        syms=symbols,  		  	   		 	 	 			  		 			 	 	 		 		 	
        allocs=allocations,  		  	   		 	 	 			  		 			 	 	 		 		 	
        sv=start_val,  		  	   		 	 	 			  		 			 	 	 		 		 	
    #   gen_plot=False,  
        gen_plot = True, 		  	   		 	 	 			  		 			 	 	 		 		 	
    )  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Print statistics  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Start Date: {start_date}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"End Date: {end_date}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Symbols: {symbols}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Allocations: {allocations}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Sharpe Ratio: {sr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Average Daily Return: {adr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Cumulative Return: {cr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	
    test_code()  		  	   		 	 	 			  		 			 	 	 		 		 	

""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""MC1-P2: Optimize a portfolio.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			 	 	 		 		 	
or edited.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Student Name: Umar Rafi  		  	   		 	 	 			  		 			 	 	 		 		 	
GT User ID: urafi3	  	   		 	 	 			  		 			 	 	 		 		 	
GT ID: 904074839 		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	

# STEPS
# 1: Import necessary libraries 
# 2: Read adjusted close prices for all symbols plus SPY (benchmark)
# 3: Find portfolio weights that maximize the Sharpe Ratio given price data.
# 4: Generate a plot comparing portfolio vs SPY normalized growth.
# 5: Compute performance metrics
# 6: Return performance statistics and final portfolio value as a tuple.		  	   		 	 	 			  		 			 	 	 		 		 	


# STEP 1: Import necessary libraries
import datetime as dt  # For handling dates and times
import warnings  # For managing warning messages

# Third-party library imports
import numpy as np  # Fundamental package for numerical computing
import pandas as pd  # Data manipulation and analysis library
import matplotlib.pyplot as plt  # MATLAB-like plotting interface

# Scientific computing
from scipy.optimize import minimize  # For optimization algorithms

# From util.py file (not using plot_data)
from util import get_data

# This is the function that will be tested by the autograder  		  	   		 	 	 			  		 			 	 	 		 		 	 		  	   		 	 	 			  		 			 	 	 		 		 	
def optimize_portfolio(
    sd=dt.datetime(2008, 1, 1),     # Start date for portfolio optimization
    ed=dt.datetime(2009, 1, 1),     # End date for portfolio optimization
    syms=["GOOG", "AAPL", "GLD", "XOM"],  # List of stock symbols to optimize allocations for
    gen_plot=True,                  # Flag to generate performance plot if True
):
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	 	 			  		 			 	 	 		 		 	
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	 	 			  		 			 	 	 		 		 	
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	 	 			  		 			 	 	 		 		 	
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	 	 			  		 			 	 	 		 		 	
    statistics.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type sd: datetime  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type ed: datetime  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	 	 			  		 			 	 	 		 		 	
        symbol in the data directory)  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type syms: list  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	 	 			  		 			 	 	 		 		 	
        code with gen_plot = False.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type gen_plot: bool  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	 	 			  		 			 	 	 		 		 	
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: tuple  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  	

    # Input validation
    if len(syms) < 2:
        raise ValueError("Must provide at least 2 symbols")
    if not all(isinstance(s, str) for s in syms):
        raise ValueError("All symbols must be strings")
    if ed <= sd:
        raise ValueError("End date must be after start date")

    # STEP 2: Read adjusted close prices for all symbols plus SPY (benchmark)  	   		 	 	 			  		 			 	 	 		 		 	
    dates = pd.date_range(sd, ed)  		  	   		 	 	 			  		 			 	 	 		 		 	
    prices_all = get_data(syms, dates)  # automatically adds SPY, using util.py function  		  	   		 	 	 			  		 			 	 	 		 		 	
    prices = prices_all[syms]  # only portfolio symbols  		  	   		 	 	 			  		 			 	 	 		 		 	
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 	 	 			  		 			 	 	 		 		 	
    
    # Normalize prices
    normed_prices = prices / prices.iloc[0]

    # Get optimal allocations
    allocs = optimize_weights(normed_prices)
    
    # Calculate daily portfolio value
    alloced = normed_prices * allocs
    port_val = alloced.sum(axis=1)

    # STEP 4: Generate a plot comparing portfolio vs SPY normalized growth.
    if gen_plot:
        df_temp = pd.concat(
            [port_val/port_val.iloc[0], prices_SPY/prices_SPY.iloc[0]],
            keys=["Portfolio", "SPY"], axis=1
        )

        plot_data(
            df_temp,
            title="Daily Portfolio Value vs SPY",
            xlabel="Date",
            ylabel="Normalized Price",
            filename="Figure1.png"
    )

    
    # STEP 5: Compute performance metrics
    cr, adr, sddr, sr = compute_portfolio_stats(port_val)

    # STEP 6: Return performance statistics and final portfolio value as a tuple.
    return allocs, cr, adr, sddr, sr

def optimize_weights(normed_price_data):
    """
    Calculate optimal portfolio weights to maximize Sharpe Ratio.
    
    :param normed_price_data: DataFrame of normalized stock prices
    :return: Optimal weight allocations as numpy array
    """
    n_stocks = normed_price_data.shape[1]
    start_weights = np.full(n_stocks, 1/n_stocks)
    
    constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})
    bounds = ((0, 1),) * n_stocks
    
    def negative_sharpe(weights):
        portfolio_value = (normed_price_data * weights).sum(axis=1)
        returns = portfolio_value.pct_change().dropna()
        return - (returns.mean() / returns.std() * np.sqrt(252))
    
    solution = minimize(negative_sharpe, start_weights, 
                       method='SLSQP', bounds=bounds, constraints=constraints)
    
    return solution.x

def compute_portfolio_stats(port_val):
    """
    Compute portfolio statistics given daily portfolio values.

    :param port_val: pd.Series of daily portfolio values
    :return: tuple (cumulative return, average daily return, std dev daily return, sharpe ratio)
    """
    # Daily returns
    daily_returns = port_val.pct_change().dropna()
    
    # Cumulative return
    cumulative_return = (port_val[-1] / port_val[0]) - 1
    
    # Average daily return
    avg_daily_return = daily_returns.mean()
    
    # Standard deviation of daily return
    std_daily_return = daily_returns.std()
    
    # Sharpe ratio (assuming 252 trading days, risk-free rate = 0)
    sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252)
    
    return cumulative_return, avg_daily_return, std_daily_return, sharpe_ratio


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price", filename="Figure1.png"):
    """Save stock prices plot with title and axis labels."""
    import matplotlib.pyplot as plt
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    

def test_code():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    This function WILL NOT be called by the auto grader.  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # start_date = dt.datetime(2009, 1, 1)  		  	   		 	 	 			  		 			 	 	 		 		 	
    #end_date = dt.datetime(2010, 1, 1)  	

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)	  	   		 	 	 			  		 			 	 	 		 		 	
    symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Assess the portfolio  		  	   		 	 	 			  		 			 	 	 		 		 	
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		 	 	 			  		 			 	 	 		 		 	
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True  		  	   		 	 	 			  		 			 	 	 		 		 	
    )  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Print statistics  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Start Date: {start_date}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"End Date: {end_date}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Symbols: {symbols}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Allocations:{allocations}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Sharpe Ratio: {sr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Average Daily Return: {adr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Cumulative Return: {cr}")  		  	   		 	 	 			  		 			 	 	 		 		 	

def author():
    """
    Returns the GT username of the student.

    Returns
    -------
    str
        GT username
    """
    return "urafi3" 

def study_group():
    """
    Returns a comma separated string of GT usernames for study group members.

    Returns
    -------
    str
        Comma-separated GT usernames or single username if working alone.
    """
    return "urafi3"  

if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	
    # This code WILL NOT be called by the auto grader  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Do not assume that it will be called  		  	   		 	 	 			  		 			 	 	 		 		 	
    test_code()
""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""MC2-P1: Market simulator.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
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
GT ID: 4074839		  	   		 	 	 			  		 			 	 	 		 		 	
"""  

######################################
# This program simulates the value of an investment portfolio over time by processing 
# a list of buy/sell orders. It uses historical stock prices to execute trades, adjusts 
# for market impact (slippage) and fixed commissions, and tracks holdings and cash to 
# compute the daily portfolio value. The simulation runs from the earliest to the latest 
# date in the orders file
######################################


########################################
# STEP 1: Read and process orders file
# STEP 2: Determine date range and symbols
# STEP 3: Get price data
# STEP 4: Process each trade order
# STEP 5: Calculate holdings
# STEP 6: Compute portfolio values
#######################################
  		  	   		 	 	 			  		 			 	 	 		 		 	
import datetime as dt   # import dt for date and time objects
import numpy as np      # import numpy for numerical operations and array handling
import pandas as pd     # import pandas for data manipulation using dataframes
from util import get_data, plot_data  # import custom utility functions for data retrieval and plotting

def compute_portvals(
    orders_file="./orders/orders.csv",
    start_val=1000000,
    commission=9.95,
    impact=0.005,
):
    """
    Computes the portfolio values.
    
    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """

    # STEP 1: Read and process orders file
    orders_df = pd.read_csv(orders_file, parse_dates=True, index_col='Date') # Load the CSV into a DataFrame with dates as the index
    orders_df.sort_index(inplace=True)                                       # Sort orders chronologically
    
    # STEP 2: Determine date range and symbols
    start_date = orders_df.index.min()
    end_date = orders_df.index.max()
    symbols = list(orders_df['Symbol'].unique())    # Extract a list of unique stock symbols involved in trading
    
    # STEP 3: Get price data
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices = prices[symbols]
    prices['Cash'] = 1.0    # Add a "Cash" column with constant value 1.0
    
    # STEP 4: Process each trade order
    trades = pd.DataFrame(0, index=prices.index, columns=prices.columns)   # Initialize a trade ledger (matrix) with all values 0
    
    # Loop through each order: gets symbol, shares, type, and price on that date
    for date, order in orders_df.iterrows():
        symbol = order['Symbol']
        shares = order['Shares']
        order_type = order['Order']
        price = prices.loc[date, symbol]
        
        if order_type == 'BUY':
            adj_price = price * (1 + impact)
            trades.loc[date, symbol] += shares
            trades.loc[date, 'Cash'] -= (adj_price * shares + commission)
        elif order_type == 'SELL':
            adj_price = price * (1 - impact)
            trades.loc[date, symbol] -= shares
            trades.loc[date, 'Cash'] += (adj_price * shares - commission)
    
    # STEP 5: Calculate holdings
    holdings = trades.cumsum()
    holdings['Cash'] += start_val   # Add initial cash to the Cash column
    
    # STEP 6: Compute portfolio values
    portvals = (holdings * prices).sum(axis=1)    # Multiply holdings by respective stock prices and sums them to get total daily portfolio value
    portvals = pd.DataFrame(portvals, columns=['Portfolio Value'])   # Wrap the series into a DataFrame
    
    return portvals

def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can used to test our code  		  	   		 	 	 			  		 			 	 	 		 		 	
    # this function will not be called during the autograding.  		  	   		 	 	 			  		 			 	 	 		 		 	
  

    of = "./orders/orders-09.csv"   # Define a test order file and starting value
    sv = 1000000
    
    # Call the compute_portvals function to get portfolio value
    portvals = compute_portvals(orders_file=of, start_val=sv)

    # Ensure portvals is a series for easier printing
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]
    else:
        "warning, code did not return a DataFrame"

    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2008, 6, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2, 0.01, 0.02, 1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2, 0.01, 0.02, 1.5]
    
    # Displays performance stats like Sharpe ratio, returns, volatility, and final value
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")

def author():
    """
    Returns the GT username of the student
    :return: The GT username of the student
    :rtype: str
    """
    return "urafi3"

def study_group():
    """
    Returns a comma-separated string of GT usernames of study group members
    :return: comma-separated string of GT usernames
    :rtype: str
    """
    return "urafi3"

if __name__ == "__main__":
    test_code() 			 	 	 		  	   		 	 	 			  		 			 	 	 		 		 	

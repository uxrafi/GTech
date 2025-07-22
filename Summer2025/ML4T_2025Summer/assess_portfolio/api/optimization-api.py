optimize_portfolio(sd=datetime.datetime(2008, 1, 1, 0, 0), ed=datetime.datetime(2009, 1, 1, 0, 0), syms=[‘GOOG’, ‘AAPL’, ‘GLD’, ‘XOM’], gen_plot=False)

    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of
    floats (as a one-dimensional NumPy array) that represent the allocations to each of the equities. You can take
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and
    statistics.

    Parameters
        sd (datetime) – A datetime object that represents the start date, defaults to 1/1/2008
        ed (datetime) – A datetime object that represents the end date, defaults to 1/1/2009
        syms (list) – A list of symbols that make up the portfolio (note that your code should support any
                      symbol in the data directory)
        gen_plot (bool) – If True, create a plot named Figure1.png (or alternatively, plot.png). 
                          The autograder will always call your code with gen_plot = False.

    Returns
        A tuple containing the portfolio allocations, cumulative return, average daily returns,
        standard deviation of daily returns, and Sharpe ratio

    Return type
        tuple

test_code()

    This function WILL NOT be called by the auto grader.


author()
    Returns
        The GT username of the student

    Return type
        str

study_group()

    Returns
        A comma separated string of GT_Name of each member of your study group
        # Example: "gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone

    Return type
        str
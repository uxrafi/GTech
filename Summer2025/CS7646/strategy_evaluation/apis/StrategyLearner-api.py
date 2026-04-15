class StrategyLearner.StrategyLearner(verbose=False, impact=0.0, commission=0.0)

    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.

    Parameters
        verbose (bool) – If “verbose” is True, your code can print out information for debugging.
                         If verbose = False your code should not generate ANY output.
        impact (float) – The market impact of each transaction, defaults to 0.0
        commission (float) – The commission amount charged, defaults to 0.0

    add_evidence(symbol='IBM', sd=datetime.datetime(2008, 1, 1, 0, 0), ed=datetime.datetime(2009, 1, 1, 0, 0), sv=100000)

        Trains your strategy learner over a given time frame.

        Parameters
            symbol (str) – The stock symbol to train on
            sd (datetime) – A datetime object that represents the start date, defaults to 1/1/2008
            ed (datetime) – A datetime object that represents the end date, defaults to 1/1/2009
            sv (int) – The starting value of the portfolio


    testPolicy(symbol='IBM', sd=datetime.datetime(2009, 1, 1, 0, 0), ed=datetime.datetime(2010, 1, 1, 0, 0), sv=100000)

        Tests your learner using data outside of the training data

        Parameters
            symbol (str) – The stock symbol that you trained on on
            sd (datetime) – A datetime object that represents the start date, defaults to 1/1/2008
            ed (datetime) – A datetime object that represents the end date, defaults to 1/1/2009
            sv (int) – The starting value of the portfolio
        Returns
            A single column data frame, indexed by date representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.

        Return type
            pandas.DataFrame

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
 
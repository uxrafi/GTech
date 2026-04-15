compute_portvals(orders_file=’./orders/orders.csv’, start_val=1000000, commission=9.95, impact=0.005)

    Computes the portfolio values.

    Parameters
        orders_file (str or file object) – Path of the order file or the file object
        start_val (int) – The starting value of the portfolio
        commission (float) – The fixed amount in dollars charged for each transaction (both entry and exit)
        impact (float) – The amount the price moves against the trader compared to the historical data at each transaction

    Returns
        the result (portvals) as a single-column dataframe indexed by date, containing the value of the portfolio for each trading day in the 
        first column from start_date to end_date, inclusive.

Return type
    pandas.DataFrame


test_code()

    Helper function to test code


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
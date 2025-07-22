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
 
best_4_dt(seed=1489683273)

    Returns data that performs significantly better with DTLearner than LinRegLearner.
    The data set should include from 2 to 10 columns in X, and one column in Y.
    The data should contain from 10 (minimum) to 1000 (maximum) rows.

    Parameters
        seed (int) – The random seed for your data generation.

    Returns
        Returns x, y - data that performs significantly better with DTLearner than LinRegLearner.
                   x – A matrix of feature values 
                   y – A vector of prediction or target values

    Return type
        numpy.ndarray


best_4_lin_reg(seed=1489683273)

    Returns data that performs significantly better with LinRegLearner than DTLearner.
    The data set should include from 2 to 10 columns in X, and one column in Y.
    The data should contain from 10 (minimum) to 1000 (maximum) rows.

    Parameters
        seed (int) – The random seed for your data generation.

    Returns
        Returns x, y - data that performs significantly better with LinRegLearner than DTLearner.
                   x – A matrix of feature values
                   y – A vector of prection or target values

    Return type
        numpy.ndarray
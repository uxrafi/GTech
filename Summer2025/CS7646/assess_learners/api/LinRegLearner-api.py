class LinRegLearner.LinRegLearner(verbose=False)
""""
    This is a Linear Regression Learner. It is implemented correctly. 
    Students will not implement this leaner. It is provided for reference only.

    Parameters
        verbose (bool) – If “verbose” is True, your code can print out information for debugging.
                         If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
"""


    add_evidence(data_x, data_y)

    #   Add training data to learner

        Parameters
            data_x (numpy.ndarray) – A set of feature values used to train the learner
            data_y (numpy.ndarray) – The value we are attempting to predict given the X data


    query(points)

        Estimate a set of test points given the model we built.

        Parameters
            points (numpy.ndarray) – A numpy array with each row corresponding to a specific query.

        Returns
            The predicted result of the input data according to the trained model

        Return type
            numpy.ndarray
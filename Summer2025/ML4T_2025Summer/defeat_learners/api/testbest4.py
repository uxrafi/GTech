compare_os_rmse(learner1, learner2, x, y)

    Compares the out-of-sample root mean squared error of your LinRegLearner and DTLearner.

    NOTE: DO NOT MODIFY THIS FILE. IT IS PROVIDED FOR REFERENCE ONLY. YOU WILL NOT SUBMIT THIS FILE FOR YOUR ASSIGNMENT.

    Parameters
        learner1 (class:’LinRegLearner.LinRegLearner’) – An instance of LinRegLearner
        learner2 (class:’DTLearner.DTLearner’) – An instance of DTLearner
        x (numpy.ndarray) – X data generated from either gen_data.best_4_dt or gen_data.best_4_lin_reg
        y (numpy.ndarray) – Y data generated from either gen_data.best_4_dt or gen_data.best_4_lin_reg

    Returns
        The root mean squared error of each learner

    Return type
        tuple

test_code()

    Performs a test of your code and prints the results
class BagLearner.BagLearner(learner, kwargs = {"argument1":1, "argument2":2}, bags = 20, boost = False, verbose = False)

    This is a Bootstrap Aggregation Learner (BagLearner). You will need to properly implement this class as necessary.

    Parameters
        learner (learner) - Points to any arbitrary learner class that will be used in the BagLearner.
        kwargs            - Keyword arguments that are passed on to the learner’s constructor and they can vary according to the learner
        bags (int)        - The number of learners you should train using Bootstrap Aggregation. 
                            If boost is true, then you should implement boosting (optional implementation).
        verbose (bool)    - If “verbose” is True, your code can print out information for debugging.
                            If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.

    add_evidence(data_x, data_y)

        Add training data to learner

        Parameters
            data_x (numpy.ndarray) – A set of feature values used to train the learner
            data_y (numpy.ndarray) – The value we are attempting to predict given the X data


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
 

    query(points)

        Estimate a set of test points given the model we built.

        Parameters
            points (numpy.ndarray) – A numpy array with each row corresponding to a specific query.

        Returns
            The predicted result of the input data according to the trained model

        Return type
            numpy.ndarray
"""
template for generating data to fool learners (c) 2016 Tucker Balch
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

"""  
STEP 1: Set up the data generation framework
STEP 2: Generate data favoring linear regression
STEP 3: Generate data favoring decision trees
STEP 4: Implement required identification functions
STEP 5: Validate implementation
STEP 6: Finalize and prepare for submission
""" 


# ==============================================
# STEP 1: Set up the data generation framework
# ==============================================
import numpy as np  # import numpy for numerical operations 

# ==============================================
# STEP 2: Generate data favoring linear regression
# ==============================================
def best_4_lin_reg(seed=1489683273):
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	 	 			  		 			 	 	 		 		 	
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 	 	 			  		 			 	 	 		 		 	
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param seed: The random seed for your data generation.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type seed: int  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: numpy.ndarray  		  	   		 	 	 			  		 			 	 	 		 		 	
    """ 
    
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Configure dataset size (within required bounds)
    num_samples = 100   # Optimal for linear relationships
    num_features = 5    # Between 2-10 features
    
    # Generate independent features (no linear combinations)
    X = np.random.normal(0, 1, (num_samples, num_features))
    
    # Define true coefficients for linear relationship
    coefficients = np.array([1.5, -2.3, 0.7, 3.1, -1.2])[:num_features]
    
    # Create target with linear relationship and small noise
    Y = X.dot(coefficients) + np.random.normal(0, 0.1, num_samples)
    
    return X, Y

# ==============================================
# STEP 3: Generate data favoring decision trees
# ==============================================
def best_4_dt(seed=1489683273):
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 	 	 			  		 			 	 	 		 		 	
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 	 	 			  		 			 	 	 		 		 	
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param seed: The random seed for your data generation.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type seed: int  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: numpy.ndarray  		  	   		 	 	 			  		 			 	 	 		 		 	
    """ 
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Configure dataset size (within required bounds)
    num_samples = 1000  # Larger dataset helps trees capture complexity
    num_features = 4    # Between 2-10 features
    
    # Generate features with uniform distribution
    X = np.random.uniform(-5, 5, (num_samples, num_features))
    Y = np.zeros(num_samples)
    
    # Create complex, piecewise non-linear relationships
    for i in range(num_samples):
        if X[i, 0] < -2.5:
            Y[i] = X[i, 1]**2 + np.abs(X[i, 2]) - 2*X[i, 3]
        elif X[i, 0] < 0:
            Y[i] = np.sin(X[i, 1]) + X[i, 2]*X[i, 3]
        elif X[i, 0] < 2.5:
            Y[i] = np.sqrt(np.abs(X[i, 1])) + X[i, 2] - X[i, 3]**2
        else:
            Y[i] = X[i, 1] + np.log(np.abs(X[i, 2]) + 1) + X[i, 3]**3
    
    # Add moderate noise
    Y += np.random.normal(0, 0.5, num_samples)
    
    return X, Y

# ==============================================
# STEP 4: Implement required identification functions
# ==============================================
def author():
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		 	 	 			  		 			 	 	 		 		 	
    """ 
    
    return "urafi3"  



# ==============================================
# STEP 5: Validate implementation and 
# STEP 6: Finalize and prepare for submission
# ==============================================

    if __name__ == "__main__":
        print("Data generators ready!")
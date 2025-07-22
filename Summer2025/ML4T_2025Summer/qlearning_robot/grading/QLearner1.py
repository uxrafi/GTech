"""   
Template for implementing QLearner  (c) 2015 Tucker Balch   

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
STEPS Summary:

STEP 1: Convert position (row, col) to discrete state index  
STEP 2: Find goal position in the map  
STEP 3: Find robot position in the map  
STEP 4: Move robot based on an action and return reward  
STEP 5: Print the current map  
STEP 6: Run the Q-Learning test loop over multiple epochs  
STEP 7: Initialize QLearner parameters and Q-table  
STEP 8: Choose action for current state (epsilon-greedy)  
STEP 9: Update Q-table with experience, perform Dyna-Q planning, and choose next action  
"""

import random as rand  
import numpy as np  

def discretize(pos):

# STEP 1: Convert the position from (row, col) to a unique state index
    return pos[0] * 10 + pos[1]  # assuming max 10 columns for simplicity

def getgoalpos(data):

# STEP 2: Locate the goal in the map
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] == 3:
                return (i, j)
    return None

def getrobotpos(data):
 
# STEP 3: Locate the robot in the map
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] == 2:
                return (i, j)
    return None

def movebot(data, oldpos, a):
    
# STEP 4: Move the robot based on the action
# Actions: 0 = up, 1 = right, 2 = down, 3 = left   
    i, j = oldpos
    data[i][j] = 0
    if a == 0 and i > 0: i -= 1
    elif a == 1 and j < len(data[0]) - 1: j += 1
    elif a == 2 and i < len(data) - 1: i += 1
    elif a == 3 and j > 0: j -= 1

    reward = -1
    if data[i][j] == 1:
        reward = -10
    elif data[i][j] == 3:
        reward = 100

    data[i][j] = 2
    return (i, j), reward

def printmap(data):

# STEP 5: Print the current map
    for row in data:
        print(" ".join(map(str, row)))
    print()

def test(map, epochs, learner, verbose):
  
# STEP 6: Run the Q-learning algorithm for a number of epochs
    total_reward = 0
    for _ in range(epochs):
        data = np.copy(map)
        robot_pos = getrobotpos(data)
        goal_pos = getgoalpos(data)
        s = discretize(robot_pos)
        a = learner.querysetstate(s)
        rsum = 0
        while robot_pos != goal_pos:
            newpos, r = movebot(data, robot_pos, a)
            s_prime = discretize(newpos)
            a = learner.query(s_prime, r)
            rsum += r
            robot_pos = newpos
            if verbose:
                printmap(data)
        total_reward += rsum
    return np.float64(total_reward)

class QLearner(object):  
    def __init__(self,  
                 num_states=100,  
                 num_actions=4,  
                 alpha=0.2,  
                 gamma=0.9,  
                 rar=0.5,  
                 radr=0.99,  
                 dyna=0,  
                 verbose=False):
        
# STEP 7: Initialize the QLearner parameters and Q-table        
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha        # learning rate
        self.gamma = gamma        # discount factor
        self.rar = rar            # random action rate (epsilon)
        self.radr = radr          # decay rate of rar
        self.dyna = dyna          # number of Dyna-Q planning steps
        self.verbose = verbose

        self.Q = np.zeros((num_states, num_actions))  # Q-table initialized to zero
        self.experience = []  # memory for Dyna-Q experience replay

        self.s = 0  # current state
        self.a = 0  # current action

    def querysetstate(self, s):
        """
        STEP 8: Choose an action based on the current state s (epsilon-greedy)
        """
        self.s = s
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)  # explore
        else:
            action = np.argmax(self.Q[s])  # exploit
        self.a = action
        return action

    def query(self, s_prime, r):
    
    # STEP 9: Update Q-table with real experience, perform Dyna-Q planning, and choose next action
    # Update Q-table with real experience
        old_q = self.Q[self.s, self.a]
        future_q = np.max(self.Q[s_prime])
        self.Q[self.s, self.a] = (1 - self.alpha) * old_q + self.alpha * (r + self.gamma * future_q)

        # Store experience for Dyna-Q
        if self.dyna > 0:
            self.experience.append((self.s, self.a, s_prime, r))

            # Perform Dyna-Q planning steps
            for _ in range(self.dyna):
                s_rand, a_rand, s_prime_rand, r_rand = rand.choice(self.experience)
                old_q_dyna = self.Q[s_rand, a_rand]
                future_q_dyna = np.max(self.Q[s_prime_rand])
                self.Q[s_rand, a_rand] = (1 - self.alpha) * old_q_dyna + self.alpha * (r_rand + self.gamma * future_q_dyna)

        # Choose next action (epsilon-greedy)
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime])

        # Decay epsilon
        self.rar *= self.radr

        # Update state and action
        self.s = s_prime
        self.a = action
        return action

    def author(self):
        return "urafi3"

    def study_group(self):
        return "urafi3"

if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")

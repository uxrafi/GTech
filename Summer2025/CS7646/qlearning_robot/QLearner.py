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

STEP 1: Convert the (row, col) position to a single state index
STEP 2: Search the map to find the goal position marked with 3
STEP 3: Search the map to find the robot's position marked with 2
STEP 4: Move the robot in the given direction and return new position and reward
STEP 5: Print the current state of the map
STEP 6: Run the learner through multiple epochs of trying to reach the goal
STEP 7: Initialize learner parameters and Q-table
STEP 8: Choose an action from the current state s using epsilon-greedy strategy
STEP 9: Update the Q-table with the experience, optionally perform Dyna-Q, and choose next action

"""

import random as rand  # for random number generation
import numpy as np  # for numerical operations

def discretize(pos):
    # STEP 1: Convert the (row, col) position to a single state index.
    return pos[0] * 10 + pos[1]  # assuming max 10 columns

def getgoalpos(data):
    # STEP 2: Search the map to find the goal position marked with 3
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] == 3:
                return (i, j)
    return None

def getrobotpos(data):
    # STEP 3: Search the map to find the robot's position marked with 2
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] == 2:
                return (i, j)
    return None

def movebot(data, oldpos, a):
    # STEP 4: Move the robot in the given direction and return new position and reward
    # Actions: 0 = up, 1 = right, 2 = down, 3 = left
    i, j = oldpos
    data[i][j] = 0  # clear current robot position

    if a == 0 and i > 0: i -= 1
    elif a == 1 and j < len(data[0]) - 1: j += 1
    elif a == 2 and i < len(data) - 1: i += 1
    elif a == 3 and j > 0: j -= 1

    reward = -1  # default movement penalty
    if data[i][j] == 1:
        reward = -10  # penalty for hitting wall
    elif data[i][j] == 3:
        reward = 100  # reward for reaching goal

    data[i][j] = 2  # place robot in new position
    return (i, j), reward

def printmap(data):
    # STEP 5: Print the current state of the map
    for row in data:
        print(" ".join(map(str, row)))
    print()

def test(map, epochs, learner, verbose):
    # STEP 6: Run the learner through multiple epochs of trying to reach the goal
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
        # STEP 7: Initialize learner parameters and Q-table
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha        # learning rate
        self.gamma = gamma        # discount factor
        self.rar = rar            # random action rate (exploration)
        self.radr = radr          # decay rate of rar
        self.dyna = dyna          # number of Dyna-Q planning steps
        self.verbose = verbose

        self.Q = np.zeros((num_states, num_actions))  # Q-table initialized with zeros
        self.experience = []  # memory to store past experiences for Dyna-Q

        self.s = 0  # current state
        self.a = 0  # current action

    def querysetstate(self, s):
        """
        STEP 8: Choose an action from the current state s using epsilon-greedy strategy
        """
        self.s = s
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)  # exploration
        else:
            action = np.argmax(self.Q[s])  # exploitation
        self.a = action
        return action

    def query(self, s_prime, r):
        """
        STEP 9: Update the Q-table with the experience, optionally perform Dyna-Q, and choose next action
        """
        # Update Q-table using the learning rule
        old_q = self.Q[self.s, self.a]
        future_q = np.max(self.Q[s_prime])
        self.Q[self.s, self.a] = (1 - self.alpha) * old_q + self.alpha * (r + self.gamma * future_q)

        # Save experience for Dyna-Q replay
        if self.dyna > 0:
            self.experience.append((self.s, self.a, s_prime, r))

            # Perform planning steps using stored experiences
            for _ in range(self.dyna):
                s_rand, a_rand, s_prime_rand, r_rand = rand.choice(self.experience)
                old_q_dyna = self.Q[s_rand, a_rand]
                future_q_dyna = np.max(self.Q[s_prime_rand])
                self.Q[s_rand, a_rand] = (1 - self.alpha) * old_q_dyna + self.alpha * (r_rand + self.gamma * future_q_dyna)

        # Decide next action
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime])

        self.rar *= self.radr  # decay exploration rate

        # Update current state and action
        self.s = s_prime
        self.a = action
        return action

    def author(self):
        return "urafi3"

    def study_group(self):
        return "urafi3"

if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")

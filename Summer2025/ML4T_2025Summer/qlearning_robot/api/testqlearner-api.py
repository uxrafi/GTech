# You do not need to modify this file. This is provided on the API Specificaion page for reference only
#

discretize(pos)

    convert the location to a single integer

    Parameters
        pos (int, int) – the position to discretize

    Returns
        the discretized position

    Return type
        int

getgoalpos(data)

    find where the goal is in the map

    Parameters
        data (array) – 2D array that stores the map

    Returns
        the position of the goal

    Return type
        tuple(int, int)

getrobotpos(data)

    Finds where the robot is in the map

    Parameters
        data (array) – 2D array that stores the map

    Returns
        the position of the robot

    Return type
        int, int

movebot(data, oldpos, a)

    move the robot and report reward

    Parameters
        data (array) – 2D array that stores the map
        oldpos (int, int) – old position of the robot
        a (int) – the action to take

    Returns
        the new position of the robot and the reward

    Return type
        tuple(int, int), int

printmap(data)

    Prints out the map

    Parameters
        data (array) – 2D array that stores the map

test(map, epochs, learner, verbose)

    function to test the code

    Parameters
        map (array) – 2D array that stores the map
        epochs (int) – each epoch involves one trip to the goal
        learner (QLearner) – the qlearner object
        verbose (bool) – If “verbose” is True, your code can print out information for debugging.
                         If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.

    Returns
        the total reward

    Return type
        np.float64
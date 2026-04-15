""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""Assess a betting strategy.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
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
                               MARTINGALE ROULETTE SIMULATION 

This project simulates a simple gambling strategy based on Professor Balch's actual betting approach using 
the Martingale system. The simulation runs 1000-bet episodes on an American roulette wheel, repeatedly doubling 
the bet after losses until a win occurs or a target profit of $80 is reached. Results are obtained using the 
get_spin_result(win_prob) function to model spin outcomes.


OBJECTIVE:

Simulate and evaluate the Martingale betting strategy over multiple episodes using
an American roulette wheel to understand the risks and expected returns of the system.


PSEUDOCODE:

episode_winnings = $0
while episode_winnings < $80:
    won = False
    bet_amount = $1
    while not won
        wager bet_amount on black
        won = result of roulette wheel spin
        if won == True:
            episode_winnings = episode_winnings + bet_amount
        else:
            episode_winnings = episode_winnings - bet_amount
            bet_amount = bet_amount * 2

STEPS:

# 1. Import required libraries
# 2. Define student metadata functions 
# 3. Define function `get_spin_result()` to simulate roulette spin based on winning probability
# 4. Define `simulate_episode()`:
#    a. Initialize winnings array and betting variables
#    b. Run spins up to max_spins or until target winnings are met
#    c. Apply Martingale logic: double bet after loss, reset after win
#    d. Record cumulative winnings per spin
# 5. Define `simulate_episode_realistic()`:
#    a. Same as above, but with additional bankroll constraint
#    b. Stops if bankroll is exhausted
# 6. Define `calculate_expectation()`:
#    a. Calculate expected value of final winnings from simulation
#    b. Use weighted probability of unique ending values
# 7. Define `test_code()` to run simulations and generate figures:
#    a. Set random seed based on GT ID for reproducibility
#    b. Simulate multiple episodes of betting strategy
#    c. Plot 10 example episodes showing first 300 spins
#    d. Plot mean winnings across episodes with standard deviation shading
#    e. Save visualizations as images
# 8. Final Results Summary and Expectation Calculation
# 9: Entry point for the simulation script
"""


#************************************************************#


# STEP 1: Import required libraries

import numpy as np               # Numerical computing library
import matplotlib                # Plotting library base module
matplotlib.use('Agg')            # Set non-interactive backend for saving plots
import matplotlib.pyplot as plt  # Plotting interface


# STEP 2: Define student metadata functions (author and GT ID)

def author():
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		 
    return "urafi3"

def study_group():
    """
    Returns comma-separated GT usernames of study group members
    Example: "urafi3, jsmith42" or "urafi3" if working alone
    """
    return "urafi3"  # Workign alone on this assignment

def gtid():
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The GT ID of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: int  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  	
    return 904074839


# STEP 3: Define function `get_spin_result()` to simulate roulette spin based on winning probability

def get_spin_result(win_prob):
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param win_prob: The probability of winning  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type win_prob: float  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The result of the spin.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: bool  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  	
    result = False  		  	   		 	 	 			  		 			 	 	 		 		 	
    if np.random.random() <= win_prob:  		  	   		 	 	 			  		 			 	 	 		 		 	
        result = True  		  	   		 	 	 			  		 			 	 	 		 		 	
    return result  


# STEP 4: Simulates spins with Martingale betting, tracking cumulative winnings until max spins or target reached

def simulate_episode(max_spins=1000, target=80):
    """Simulation with proper array structure including initial value

    Parameters:
        max_spins (int): Maximum number of spins to simulate (default 1000)
        target (int): Profit target in dollars that will stop the episode (default $80)
    
    Returns:
        np.ndarray: Array of cumulative winnings where:
                   - index 0 represents initial state (always 0)
                   - index 1 represents after 1st spin
                   - index n represents after nth spin
    """

    # Initialize winnings array with max_spins+1 elements to include starting point (spin 0)
    winnings = np.zeros(max_spins + 1)  # Now includes initial value at index 0
    # Initialize tracking variables
    current_winnings = 0
    bet_amount = 1
    
    # Main simulation loop for each spin (starting at spin 1)
    for spin in range(1, max_spins + 1):
        if current_winnings >= target:
            winnings[spin:] = current_winnings  # Fill remaining spins with current value
            break
            
        won = get_spin_result(18/38)  # American roulette probability
        
        if won:  # Winning scenario:
            current_winnings += bet_amount
            bet_amount = 1  # Reset after win
        else: # Losing scenario:
            current_winnings -= bet_amount
            bet_amount *= 2  # Double after loss
        
        # Record current winnings after this spin
        winnings[spin] = current_winnings
    
    return winnings


# STEP 5: Simulates spins with Martingale betting and bankroll limit, stopping if bankroll exhausted or target reached

def simulate_episode_realistic(max_spins=1000, target=80, bankroll=256):
    """Realistic simulation with bankroll constraints and proper array structure"""
    winnings = np.zeros(max_spins + 1)  # Includes initial value
    current_winnings = 0
    bet_amount = 1
    consecutive_losses = 0
    
    for spin in range(1, max_spins + 1):
        if current_winnings >= target or current_winnings <= -bankroll:
            winnings[spin:] = current_winnings
            break
            
        # Calculate maximum possible bet considering bankroll
        remaining_cash = bankroll + current_winnings
        bet_amount = min(2**consecutive_losses, remaining_cash)
        
        won = get_spin_result(18/38)
        if won:
            current_winnings += bet_amount
            consecutive_losses = 0
            bet_amount = 1
        else:
            current_winnings -= bet_amount
            consecutive_losses += 1
        
        winnings[spin] = current_winnings
    
    return winnings


# STEP 6: Calculates expected final winnings from simulation outcomes using weighted probabilities

def calculate_expectation(results):
    
    """
    Calculates probability-weighted expectation of final winnings
    
    This computes the mathematical expectation (mean outcome) by considering each unique
    ending value and its relative frequency in the simulation results.
    
    Parameters:
        results (np.ndarray): 2D array of simulation results where:
                            - Each row represents one episode
                            - Last column (results[:,-1]) contains final winnings
    
    Returns:
        float: Expected value (probability-weighted average) of final winnings
    """
    # Extract all unique final winnings values and their counts
    # results[:,-1] gets the last element of each episode (final winnings)
    unique, counts = np.unique(results[:,-1], return_counts=True)

    # Convert raw counts to probabilities by dividing by total number of episodes
    probabilities = counts / counts.sum()

    # Calculate expectation as the weighted sum of outcomes
    # E[X] = Σ(x_i * P(x_i)) for all unique outcomes x_i
    expected_value = np.sum(unique * probabilities)
    
    return expected_value


# STEP 7: Experiment and Visualization Functions

def run_original_experiment(num_episodes=1000, max_spins=1000):
    """
    Run the original Martingale strategy simulation
    Returns: 2D numpy array of results (episodes x spins)
    """
    return np.array([simulate_episode(max_spins) for _ in range(num_episodes)])

def run_realistic_experiment(num_episodes=1000, max_spins=1000):
    """
    Run the realistic Martingale strategy simulation with bankroll constraint
    Returns: 2D numpy array of results (episodes x spins)
    """
    return np.array([simulate_episode_realistic(max_spins) for _ in range(num_episodes)])

def generate_episode_plot(results, filename="figure1.png"):
    """Plot 10 unlimited-money episodes"""
    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i in range(10):
        plt.plot(results[i,:301], color=colors[i], label=f'Episode {i+1}')
    
    plt.title("Martingale Strategy (Unlimited Bankroll)\n10 Example Episodes", pad=15)
    plt.xlabel("Bet Number")
    plt.ylabel("Winnings ($)")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend(bbox_to_anchor=(1.02, 1))
    plt.grid(alpha=0.3)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def generate_mean_plot(results, filename="figure2.png"):
    """Generate plot showing mean winnings with std dev"""
    plt.figure(figsize=(10, 6))
    plt.plot(np.mean(results, axis=0)[:301], label="Mean")
    plt.fill_between(range(301),
                   np.mean(results, axis=0)[:301] - np.std(results, axis=0)[:301],
                   np.mean(results, axis=0)[:301] + np.std(results, axis=0)[:301],
                   alpha=0.2, label="±1 Std Dev")
    plt.title("Mean Winnings of 1000 Episodes")
    plt.xlabel("Number of Bets")
    plt.ylabel("Winnings ($)")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def generate_median_plot(results, filename="figure3.png"):
    """Generate plot showing median winnings with std dev"""
    plt.figure(figsize=(10, 6))
    plt.plot(np.median(results, axis=0)[:301], label="Median")
    plt.fill_between(range(301),
                   np.median(results, axis=0)[:301] - np.std(results, axis=0)[:301],
                   np.median(results, axis=0)[:301] + np.std(results, axis=0)[:301],
                   alpha=0.2, label="±1 Std Dev")
    plt.title("Median Winnings of 1000 Episodes")
    plt.xlabel("Number of Bets")
    plt.ylabel("Winnings ($)")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_simulation_results(results_original, results_realistic, filename="p1_results.txt"):
    """Save experiment results to file"""
    # Open file in write mode (creates or overwrites)
    with open(filename, "w") as f:
        
        # Write header section
        f.write("=== Experiment Results ===\n")
        
        # Original strategy success rate (% episodes reaching >= $80)
        f.write(f"Original Strategy Success Rate: {np.mean([np.max(ep) >= 80 for ep in results_original]):.2%}\n")
        
        # Original strategy average final winnings
        f.write(f"Original Strategy Avg Final Winnings: ${np.mean(results_original[:,-1]):.2f}\n")
        
        # Original strategy probability-weighted expectation
        f.write(f"Original Strategy Expectation: ${calculate_expectation(results_original):.2f}\n\n")
        
        # Realistic strategy success rate (% episodes reaching >= $80)
        f.write(f"Realistic Strategy Success Rate: {np.mean([np.max(ep) >= 80 for ep in results_realistic]):.2%}\n")
        
        # Realistic strategy bankruptcy rate (% episodes reaching <= -$256)
        f.write(f"Realistic Strategy Bankruptcy Rate: {np.mean([np.min(ep) <= -256 for ep in results_realistic]):.2%}\n")
        
        # Realistic strategy average final winnings
        f.write(f"Realistic Strategy Avg Final Winnings: ${np.mean(results_realistic[:,-1]):.2f}\n")
        
        # Realistic strategy probability-weighted expectation
        f.write(f"Realistic Strategy Expectation: ${calculate_expectation(results_realistic):.2f}\n")


# STEP 8: Main test function

def test_code():
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    Method to test your code  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  
    np.random.seed(gtid())
    
    # Run experiments
    original_results = run_original_experiment()
    realistic_results = run_realistic_experiment()
    
    # Generate figures for original strategy
    generate_episode_plot(original_results, "figure1.png")
    generate_mean_plot(original_results, "figure2.png")
    generate_median_plot(original_results, "figure3.png")
    
    # Generate figures for realistic strategy
    generate_mean_plot(realistic_results, "figure4.png")
    generate_median_plot(realistic_results, "figure5.png")
    
    # Save results
    save_simulation_results(original_results, realistic_results)


# STEP 9: Entry point

if __name__ == "__main__":
    test_code()
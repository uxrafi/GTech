"""
STEP 1: Load and preprocess the dataset
STEP 2: Run Experiment 1 - Analyze overfitting in DTLearner
STEP 3: Run Experiment 2 - Compare single DT vs Bagged DTs
STEP 4: Run Experiment 3 - Compare DTLearner vs RTLearner on performance
STEP 5: Save generated plots and statistics to files
"""

import sys              # For handling command-line arguments
import math             # For mathematical functions (sqrt, etc.)
import time             # To measure training and query durations
import tracemalloc      # To track memory usage of learners
import numpy as np      # For handling numerical arrays and matrix operations
import matplotlib.pyplot as plt  # For creating and saving plots

import DTLearner as dt  # Decision Tree Learner module
import RTLearner as rt  # Random Tree Learner module
import BagLearner as bl # Bagging ensemble learner module

# ----------------------------- Metrics Implementation -----------------------------

# Root Mean Squared Error
def rmse(y_true, y_pred):
    return math.sqrt(np.mean((y_true - y_pred) ** 2))

# Mean Absolute Error (manually implemented)
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# R-squared Score (manually implemented)
def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# ----------------------------- Plotting Utility -----------------------------

# Helper function to save line plots for RMSE metrics
def save_plot(x, y_series, labels, title, xlabel, ylabel, vline=None, filename=None):
    plt.figure(figsize=(10, 6))
    for y, label in zip(y_series, labels):
        plt.plot(x, y, label=label)
    if vline:
        plt.axvline(x=vline, color='r', linestyle='--', label=f'Optimal (leaf={vline})')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.close()

# ----------------------------- STEP 1: Load Data -----------------------------

def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    if "Istanbul" in filename:
        data = data[:, 1:]  # Remove index column if present
    np.random.shuffle(data)
    split = int(0.6 * len(data))
    return data[:split, :-1], data[:split, -1], data[split:, :-1], data[split:, -1]

# ----------------------------- STEP 2: Experiment 1 -----------------------------

def run_experiment1(train_x, train_y, test_x, test_y):
    leaf_sizes = range(1, 51)
    train_rmses, test_rmses = [], []

    for leaf_size in leaf_sizes:
        learner = dt.DTLearner(leaf_size=leaf_size)
        learner.add_evidence(train_x, train_y)
        train_rmses.append(rmse(train_y, learner.query(train_x)))
        test_rmses.append(rmse(test_y, learner.query(test_x)))

    optimal_leaf = leaf_sizes[np.argmin(test_rmses)]

    save_plot(leaf_sizes, [train_rmses, test_rmses],
              ['Train RMSE', 'Test RMSE'],
              'Experiment 1: DTLearner Overfitting Analysis',
              'Leaf Size', 'RMSE', vline=optimal_leaf, filename='exp1_overfitting.png')

    return {
        'optimal_leaf': optimal_leaf,
        'min_rmse': test_rmses[optimal_leaf - 1],
        'train_rmses': train_rmses,
        'test_rmses': test_rmses
    }

# ----------------------------- STEP 3: Experiment 2 -----------------------------

def run_experiment2(train_x, train_y, test_x, test_y, optimal_leaf):
    leaf_range = range(max(1, optimal_leaf - 10), min(51, optimal_leaf + 11))
    reg_rmse, bag_rmse = [], []

    for leaf_size in leaf_range:
        dt_model = dt.DTLearner(leaf_size=leaf_size)
        dt_model.add_evidence(train_x, train_y)
        reg_rmse.append(rmse(test_y, dt_model.query(test_x)))

        bag_model = bl.BagLearner(learner=dt.DTLearner,
                                  kwargs={"leaf_size": leaf_size},
                                  bags=20)
        bag_model.add_evidence(train_x, train_y)
        bag_rmse.append(rmse(test_y, bag_model.query(test_x)))

    save_plot(leaf_range, [reg_rmse, bag_rmse],
              ['Regular DT', 'Bagged DT (20 bags)'],
              'Experiment 2: Bagging Effect on Overfitting',
              'Leaf Size', 'RMSE', vline=optimal_leaf, filename='exp2_bagging.png')

    return {'regular_rmses': reg_rmse, 'bagged_rmses': bag_rmse}

# ----------------------------- STEP 4: Experiment 3 -----------------------------

def run_experiment3(train_x, train_y, test_x, test_y):
    results = {learner: {'time': [], 'memory': [], 'mae': [], 'r2': []} for learner in ['DT', 'RT']}
    
    for _ in range(10):
        for name, Learner in [('DT', dt.DTLearner), ('RT', rt.RTLearner)]:
            tracemalloc.start()
            start = time.time()
            model = Learner(leaf_size=5)
            model.add_evidence(train_x, train_y)
            pred = model.query(test_x)
            results[name]['time'].append(time.time() - start)
            results[name]['memory'].append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
            results[name]['mae'].append(mean_absolute_error(test_y, pred))
            results[name]['r2'].append(r2_score(test_y, pred))

    # Create boxplots for all metrics
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ['mae', 'r2', 'time', 'memory']
    titles = ['MAE Comparison', 'R² Comparison', 'Training Time', 'Memory Usage']
    ylabels = ['Mean Absolute Error', 'R²', 'Seconds', 'Bytes']

    for ax, metric, title, ylabel in zip(axs.flat, metrics, titles, ylabels):
        ax.boxplot([results['DT'][metric], results['RT'][metric]])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['DT', 'RT'])
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig('exp3_comparison.png')
    plt.close()
    return results

# ----------------------------- STEP 5: Save Statistics -----------------------------

def save_stats(exp1, exp2, exp3):
    with open('stats.txt', 'w') as f:
        f.write("EXPERIMENT RESULTS\n" + "="*50 + "\n\n")

        f.write("EXPERIMENT 1: OVERFITTING ANALYSIS\n" + "-"*50 + "\n")
        f.write(f"Optimal leaf size: {exp1['optimal_leaf']}\n")
        f.write(f"Minimum Test RMSE: {exp1['min_rmse']:.6f}\n")
        f.write(f"Training RMSE at optimal: {exp1['train_rmses'][exp1['optimal_leaf'] - 1]:.6f}\n")
        f.write("Chart saved to: exp1_overfitting.png\n\n")

        f.write("EXPERIMENT 2: BAGGING EFFECTS\n" + "-"*50 + "\n")
        f.write(f"Avg RMSE reduction: {np.mean(exp2['regular_rmses']) - np.mean(exp2['bagged_rmses']):.6f}\n")
        f.write("Chart saved to: exp2_bagging.png\n\n")

        f.write("EXPERIMENT 3: DT vs RT COMPARISON\n" + "-"*50 + "\n")
        for metric in ['mae', 'r2', 'time', 'memory']:
            dt_avg = np.mean(exp3['DT'][metric])
            rt_avg = np.mean(exp3['RT'][metric])
            f.write(f"{metric.upper():<8} | DT: {dt_avg:.6f} | RT: {rt_avg:.6f}\n")
        f.write("Chart saved to: exp3_comparison.png\n")

# ----------------------------- Main Program Entry -----------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    try:
        print("Loading data...")
        train_x, train_y, test_x, test_y = load_data(sys.argv[1])

        print("Running Experiment 1...")
        exp1 = run_experiment1(train_x, train_y, test_x, test_y)

        print("Running Experiment 2...")
        exp2 = run_experiment2(train_x, train_y, test_x, test_y, exp1['optimal_leaf'])

        print("Running Experiment 3...")
        exp3 = run_experiment3(train_x, train_y, test_x, test_y)

        print("Saving statistics...")
        save_stats(exp1, exp2, exp3)

        print("\nEXPERIMENT COMPLETED SUCCESSFULLY!")
        print("Generated:")
        print("- exp1_overfitting.png")
        print("- exp2_bagging.png")
        print("- exp3_comparison.png")
        print("- stats.txt")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

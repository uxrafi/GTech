import numpy as np
from LinRegLearner import LinRegLearner
from DTLearner import DTLearner
from gen_data import best_4_lin_reg, best_4_dt

def evaluate_learner(X, Y, learner):
    """Train and evaluate a learner, returning RMSE."""
    learner.add_evidence(X, Y)
    pred = learner.query(X)
    return np.sqrt(((Y - pred) ** 2).mean())

def test_multiple_seeds(seeds):
    linreg_passes = 0
    dt_passes = 0

    for seed in seeds:
        print(f"\n=== Testing Seed: {seed} ===")

        # Test best_4_lin_reg
        X, Y = best_4_lin_reg(seed)
        rmse_lr = evaluate_learner(X, Y, LinRegLearner())
        rmse_dt = evaluate_learner(X, Y, DTLearner(leaf_size=1))
        lr_success = rmse_lr < 0.9 * rmse_dt
        linreg_passes += 1 if lr_success else 0
        print(f"LinReg RMSE: {rmse_lr:.4f}, DT RMSE: {rmse_dt:.4f}")
        print(f"LinReg < 0.9*DT? {'✅ PASS' if lr_success else '❌ FAIL'}")

        # Test best_4_dt
        X, Y = best_4_dt(seed)
        rmse_lr = evaluate_learner(X, Y, LinRegLearner())
        rmse_dt = evaluate_learner(X, Y, DTLearner(leaf_size=1))
        dt_success = rmse_dt < 0.9 * rmse_lr
        dt_passes += 1 if dt_success else 0
        print(f"LinReg RMSE: {rmse_lr:.4f}, DT RMSE: {rmse_dt:.4f}")
        print(f"DT < 0.9*LinReg? {'✅ PASS' if dt_success else '❌ FAIL'}")

    print("\n=== Summary ===")
    print(f"LinReg passes: {linreg_passes}/{len(seeds)}")
    print(f"DT passes: {dt_passes}/{len(seeds)}")

if __name__ == "__main__":
    # Test 15 different seeds
    seeds_to_test = [42, 123, 999, 555, 1001, 7, 13, 404, 2023, 314159, 
                     987, 654, 321, 111, 222]  # Example seeds
    test_multiple_seeds(seeds_to_test)
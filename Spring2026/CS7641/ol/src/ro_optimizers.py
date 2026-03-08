"""Randomized optimization algorithms: RHC, SA, GA with history tracking.

Design disclosures (required by assignment):
  RHC  — Gaussian step perturbation, configurable step_size; random restarts
          from U(-1,1) after each restart budget; plateau rule stops a restart
          after `plateau_patience` evals with no improvement.
  SA   — Gaussian step perturbation; exponential decay schedule T_t = T_0 * decay^t;
          acceptance rate logged every 100 evaluations; step size fixed.
  GA   — Uniform crossover; tournament selection (size 2); Gaussian mutation
          scaled by `mutation_std`; hard elitism preserves top `elitism` individuals.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Randomized Hill Climbing
# ---------------------------------------------------------------------------
class RandomizedHillClimbing:
    """RHC with restarts and plateau detection.

    Parameters
    ----------
    step_size         : std dev of Gaussian perturbation
    restarts          : number of independent restarts (including first run)
    max_evaluations   : total budget across all restarts
    plateau_patience  : stop a restart early after this many evals with no improvement
    """

    def __init__(self, step_size=0.01, restarts=3,
                 max_evaluations=5000, plateau_patience=500):
        self.step_size        = step_size
        self.restarts         = restarts
        self.max_evals        = max_evaluations
        self.plateau_patience = plateau_patience

    def optimize(self, objective, initial_params, bounds=None, return_history=False):
        best_params = initial_params.copy()
        best_loss   = objective(initial_params.copy())
        evaluations = 1
        history     = [(evaluations, best_loss)] if return_history else None

        evals_per_restart = max(1, self.max_evals // self.restarts)

        for restart in range(self.restarts):
            if restart == 0:
                current      = initial_params.copy()
                current_loss = best_loss
            else:
                current      = np.random.uniform(-1, 1, size=initial_params.shape)
                current_loss = objective(current)
                evaluations += 1
                if return_history:
                    history.append((evaluations, best_loss))

            no_improve = 0
            restart_budget = evals_per_restart

            while no_improve < self.plateau_patience and restart_budget > 0:
                if evaluations >= self.max_evals:
                    break

                neighbor = current + self.step_size * np.random.randn(*current.shape)
                if bounds is not None:
                    lo = np.array([b[0] for b in bounds])
                    hi = np.array([b[1] for b in bounds])
                    neighbor = np.clip(neighbor, lo, hi)

                neighbor_loss = objective(neighbor)
                evaluations  += 1
                restart_budget -= 1

                if neighbor_loss < current_loss:
                    current      = neighbor
                    current_loss = neighbor_loss
                    no_improve   = 0
                    if current_loss < best_loss:
                        best_params = current.copy()
                        best_loss   = current_loss
                        if return_history:
                            history.append((evaluations, best_loss))
                else:
                    no_improve += 1

            if evaluations >= self.max_evals:
                break

        if return_history:
            # Ensure final eval is recorded
            if history[-1][0] != evaluations:
                history.append((evaluations, best_loss))
            return best_params, best_loss, evaluations, history
        return best_params, best_loss, evaluations


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------
class SimulatedAnnealing:
    """SA with exponential temperature decay and acceptance-rate logging.

    Parameters
    ----------
    initial_temp     : T_0 — starting temperature
    decay            : multiplicative decay per step (T_{t+1} = T_t * decay)
    step_size        : std dev of Gaussian perturbation
    max_evaluations  : total budget
    log_interval     : log acceptance stats every this many evaluations
    """

    def __init__(self, initial_temp=1.0, decay=0.99, step_size=0.01,
                 max_evaluations=5000, log_interval=100):
        self.T0           = initial_temp
        self.decay        = decay
        self.step_size    = step_size
        self.max_evals    = max_evaluations
        self.log_interval = log_interval

    def optimize(self, objective, initial_params, bounds=None, return_history=False):
        current      = initial_params.copy()
        current_loss = objective(current)
        best         = current.copy()
        best_loss    = current_loss
        evaluations  = 1
        T            = self.T0
        history      = [(evaluations, best_loss)] if return_history else None

        # Acceptance-rate tracking
        self.acceptance_log = []   # list of (eval, acceptance_rate_in_window)
        window_accepts = 0
        window_total   = 0

        while evaluations < self.max_evals:
            neighbor = current + self.step_size * np.random.randn(*current.shape)
            if bounds is not None:
                lo = np.array([b[0] for b in bounds])
                hi = np.array([b[1] for b in bounds])
                neighbor = np.clip(neighbor, lo, hi)

            neighbor_loss = objective(neighbor)
            evaluations  += 1
            window_total += 1

            delta = neighbor_loss - current_loss
            # Guard against T -> 0 division
            if T > 1e-10:
                accept_prob = np.exp(-delta / T) if delta >= 0 else 1.0
            else:
                accept_prob = 1.0 if delta < 0 else 0.0

            accepted = np.random.rand() < accept_prob
            if accepted:
                current      = neighbor
                current_loss = neighbor_loss
                window_accepts += 1
                if current_loss < best_loss:
                    best      = current.copy()
                    best_loss = current_loss
                    if return_history:
                        history.append((evaluations, best_loss))

            T *= self.decay

            # Log acceptance rate in this window
            if window_total >= self.log_interval:
                rate = window_accepts / window_total
                self.acceptance_log.append((evaluations, rate))
                window_accepts = 0
                window_total   = 0

        if return_history:
            if history[-1][0] != evaluations:
                history.append((evaluations, best_loss))
            return best, best_loss, evaluations, history
        return best, best_loss, evaluations


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------
class GeneticAlgorithm:
    """GA with tournament selection, uniform crossover, Gaussian mutation, elitism.

    Parameters
    ----------
    pop_size        : population size (compute budget per generation = pop_size evals)
    mutation_rate   : probability of mutating a child
    mutation_std    : std dev of Gaussian mutation noise (applied per-gene)
    crossover_rate  : probability of producing a crossover child vs. a clone
    elitism         : number of best individuals carried over unchanged each generation
    max_evaluations : total function-evaluation budget
    """

    def __init__(self, pop_size=30, mutation_rate=0.1, mutation_std=0.01,
                 crossover_rate=0.8, elitism=2, max_evaluations=5000):
        self.pop_size      = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_std  = mutation_std
        self.crossover_rate = crossover_rate
        self.elitism       = elitism
        self.max_evals     = max_evaluations

    def _tournament(self, population, fitness):
        """Binary tournament selection — returns one parent."""
        i1, i2 = np.random.randint(self.pop_size, size=2)
        return population[i1] if fitness[i1] < fitness[i2] else population[i2]

    def optimize(self, objective, initial_params, bounds=None, return_history=False):
        dim = len(initial_params)
        lo  = np.array([b[0] for b in bounds]) if bounds else None
        hi  = np.array([b[1] for b in bounds]) if bounds else None

        # Initialise: perturb initial_params with small Gaussian noise
        population = [initial_params + 0.1 * np.random.randn(dim)
                      for _ in range(self.pop_size)]
        if bounds is not None:
            population = [np.clip(p, lo, hi) for p in population]

        fitness     = [objective(ind) for ind in population]
        evaluations = self.pop_size

        best_idx  = int(np.argmin(fitness))
        best      = population[best_idx].copy()
        best_loss = fitness[best_idx]
        history   = [(evaluations, best_loss)] if return_history else None

        while evaluations < self.max_evals:
            sorted_idx  = np.argsort(fitness)
            new_pop = [population[i].copy() for i in sorted_idx[:self.elitism]]

            while len(new_pop) < self.pop_size:
                if np.random.rand() < self.crossover_rate:
                    p1 = self._tournament(population, fitness)
                    p2 = self._tournament(population, fitness)
                    mask  = np.random.rand(dim) < 0.5
                    child = np.where(mask, p1, p2).copy()
                else:
                    child = population[np.random.randint(self.pop_size)].copy()

                if np.random.rand() < self.mutation_rate:
                    child += self.mutation_std * np.random.randn(dim)

                if bounds is not None:
                    child = np.clip(child, lo, hi)
                new_pop.append(child)

            population  = new_pop
            fitness     = [objective(ind) for ind in population]
            evaluations += self.pop_size

            gen_best = int(np.argmin(fitness))
            if fitness[gen_best] < best_loss:
                best      = population[gen_best].copy()
                best_loss = fitness[gen_best]
                if return_history:
                    history.append((evaluations, best_loss))

            if evaluations >= self.max_evals:
                break

        if return_history:
            if history[-1][0] != evaluations:
                history.append((evaluations, best_loss))
            return best, best_loss, evaluations, history
        return best, best_loss, evaluations
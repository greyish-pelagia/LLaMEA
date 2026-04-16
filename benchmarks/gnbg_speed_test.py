import time
from functools import wraps

import iohgnbg
import numpy as np
from ioh import get_problem, logger
from modcma import c_maes

from benchmarks.gnbg.loader import make_problem
from misc import OverBudgetException, aoc_logger, correct_aoc


def time_iterations():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            print(f"Execution of {func.__name__} took {execution_time:.4f}s")

            return execution_time

        return wrapper

    return decorator


@time_iterations()
def evaluate_gnbg(budget_mul: int = 2000):

    problems = iohgnbg.get_problems(
        problem_indices=24,
        instances_folder="benchmarks/gnbg/official",  # change the problem instances .mat files by specifying the dir name with them
    )

    aucs = []

    l2 = aoc_logger(budget_mul, upper=1e2, triggers=[logger.trigger.ALWAYS])

    for problem in problems:
        problem.attach_logger(l2)
        x0 = np.zeros(shape=(problem.meta_data.n_variables))

        lb = problem.bounds.lb[0]
        ub = problem.bounds.ub[0]
        sigma0 = 0.3 * (ub - lb)
        budget = problem.meta_data.n_variables * budget_mul

        try:
            c_maes.fmin(
                problem,
                x0,
                sigma0,
                budget,
            )
        except OverBudgetException:
            pass

        auc = correct_aoc(problem, l2, budget)
        aucs.append(auc)
        l2.reset(problem)
        problem.reset()

    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)

    print(
        f"CMA-ES got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.4f} with standard deviation {auc_std:0.4f}."
    )


@time_iterations()
def evaluate_bbob(budget_mul: int = 2000):

    dim = 30

    aucs = []

    l2 = aoc_logger(budget_mul, upper=1e2, triggers=[logger.trigger.ALWAYS])

    for fid in np.arange(1, 25):
        for iid in [1]:  # , 4, 5]
            problem = get_problem(fid, iid, dim)
            problem.attach_logger(12)

            x0 = np.zeros(shape=(problem.meta_data.n_variables))

            lb = problem.bounds.lb[0]
            ub = problem.bounds.ub[0]
            sigma0 = 0.3 * (ub - lb)
            budget = problem.meta_data.n_variables * budget_mul

            try:
                c_maes.fmin(
                    problem,
                    x0,
                    sigma0,
                    budget,
                )
            except OverBudgetException:
                pass

            auc = correct_aoc(problem, l2, budget)
            aucs.append(auc)
            l2.reset(problem)
            problem.reset()

        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)

        print(
            f"CMA-ES got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.4f} with standard deviation {auc_std:0.4f}."
        )


@time_iterations()
def evaluate_pytgnbg(budget_mul: int = 2000):
    run_scores = []
    run_details = []

    for fid in range(1, 25):
        for rep in range(1):
            np.random.seed(rep)
            problem = make_problem(fid)

            x0 = np.zeros(shape=(problem.dim))

            lb = problem.lower
            ub = problem.upper
            sigma0 = 0.3 * (ub - lb)
            budget = problem.dim * budget_mul

            try:
                c_maes.fmin(
                    problem,
                    x0,
                    sigma0,
                    budget,
                )
            except OverBudgetException:
                pass

            if np.isfinite(problem.best_y):
                err = abs(problem.best_y - problem.optimum)
                score = -np.log10(max(err, 1e-12))
            else:
                err = float("inf")
                score = float("-inf")

            run_scores.append(score)
            run_details.append(
                {
                    "fid": fid,
                    "rep": rep,
                    "best_y": problem.best_y,
                    "err": err,
                    "fes": problem.evaluations,
                    "arp": problem.acceptance_reach_point,
                    "best_x": None
                    if problem.best_x is None
                    else problem.best_x.tolist(),
                }
            )

    score_mean = float(np.mean(run_scores))
    score_std = float(np.std(run_scores))

    print(
        f"CMA-ES got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {score_mean:0.4f} with standard deviation {score_std:0.4f}."
    )


if __name__ == "__main__":
    evaluate_gnbg(2000)
    evaluate_pytgnbg(2000)
    # evaluate_bbob(2000)

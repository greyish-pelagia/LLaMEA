import time
from functools import wraps

import iohgnbg
import numpy as np
from ioh import get_problem, logger
from modcma import c_maes

from misc import OverBudgetException, aoc_logger, correct_aoc


def time_iterations():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            print(f"Execution of {func.__name__} took {execution_time}s")

            return execution_time

        return wrapper

    return decorator


@time_iterations()
def evaluate_gnbg(budget_mul: int = 2000):

    problems = iohgnbg.get_problems(problem_indices=24)

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


if __name__ == "__main__":
    evaluate_gnbg(2000)
    # evaluate_bbob(2000)

import time
from functools import wraps

import iohgnbg
import numpy as np
from gnbg_speed_test import time_iterations
from ioh import get_problem, logger
from modcma import c_maes

from misc import OverBudgetException, aoc_logger, correct_aoc


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


if __name__ == "__main__":
    evaluate_gnbg(2000)

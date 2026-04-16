import json

import iohgnbg
import numpy as np
from ioh import logger
from modcma import c_maes

from benchmarks.gnbg_speed_test import time_iterations
from misc import OverBudgetException, aoc_logger, correct_aoc


def run_gnbg_benchmark(optimizer_fn, name: str, budget_mul: int = 2000, debug: int = 0):
    """Generic runner for GNBG problems"""
    problems = iohgnbg.get_problems(
        problem_indices=24,
        instances_folder="benchmarks/gnbg/official",
    )

    aucs = []
    debug_dict = {}
    l2 = aoc_logger(budget_mul, upper=1e2, triggers=[logger.trigger.ALWAYS])

    for problem in problems:
        problem.attach_logger(l2)
        dim = problem.meta_data.n_variables
        budget = dim * budget_mul

        try:
            optimizer_fn(problem, dim, budget)
        except OverBudgetException:
            pass

        auc = correct_aoc(problem, l2, budget)
        aucs.append(auc)

        debug_dict[problem.meta_data.name] = {
            "Num evaluations": problem.state.evaluations,
            "Final target found": problem.state.final_target_found,
            "Current best": problem.state.current_best.y,
            "Problem optimum": problem.optimum.y,
            "AUC": auc,
        }
        l2.reset(problem)
        problem.reset()

    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)
    print(f"{name} average AOCC: {auc_mean:0.4f} (std: {auc_std:0.4f})")

    if debug == 10:
        top_3 = dict(sorted(debug_dict.items(), key=lambda item: item[1]["AUC"])[:3])
        print(json.dumps(top_3, indent=4))
    elif debug == 20:
        print(json.dumps(debug_dict, indent=4))


@time_iterations()
def cmaes_gnbg(budget_mul: int = 2000, debug: int = 0):
    def cmaes_logic(problem, dim, budget):
        lb, ub = problem.bounds.lb[0], problem.bounds.ub[0]
        x0 = np.random.uniform(lb, ub, size=dim)
        sigma0 = 0.3 * (ub - lb)
        c_maes.fmin(problem, x0, sigma0, budget)

    run_gnbg_benchmark(cmaes_logic, "CMA-ES", budget_mul, debug)


@time_iterations()
def bipop_cmaes_gnbg(budget_mul: int = 2000, debug: int = 0):
    modules = c_maes.parameters.Modules()
    modules.active = True
    modules.restart_strategy = c_maes.options.RestartStrategy.BIPOP

    def bipop_logic(problem, dim, budget):
        lb, ub = problem.bounds.lb[0], problem.bounds.ub[0]
        x0 = np.random.uniform(lb, ub, size=dim)
        settings = c_maes.parameters.Settings(
            dim=dim,
            modules=modules,
            x0=x0,
            sigma0=0.3 * (ub - lb),
            budget=budget,
            lb=problem.bounds.lb,
            ub=problem.bounds.ub,
        )
        cma = c_maes.ModularCMAES(c_maes.Parameters(settings))
        cma.run(problem)

    run_gnbg_benchmark(bipop_logic, "BIPOP-CMA-ES", budget_mul, debug)


@time_iterations()
def ipop_cmaes_gnbg(budget_mul: int = 2000, debug: int = 0):
    modules = c_maes.parameters.Modules()
    modules.active = False
    modules.restart_strategy = c_maes.options.RestartStrategy.IPOP

    def logic(problem, dim, budget):
        lb, ub = problem.bounds.lb[0], problem.bounds.ub[0]
        x0 = np.random.uniform(lb, ub, size=dim)
        settings = c_maes.parameters.Settings(
            dim=dim,
            modules=modules,
            x0=x0,
            sigma0=0.3 * (ub - lb),
            budget=budget,
            lb=problem.bounds.lb,
            ub=problem.bounds.ub,
        )
        cma = c_maes.ModularCMAES(c_maes.Parameters(settings))
        cma.run(problem)

    run_gnbg_benchmark(logic, "IPOP-CMA-ES", budget_mul, debug)


@time_iterations()
def mirrored_cmaes_gnbg(budget_mul: int = 2000, debug: int = 0):
    modules = c_maes.parameters.Modules()
    modules.mirrored = c_maes.options.Mirror.MIRRORED

    def logic(problem, dim, budget):
        lb, ub = problem.bounds.lb[0], problem.bounds.ub[0]
        x0 = np.random.uniform(lb, ub, size=dim)
        settings = c_maes.parameters.Settings(
            dim=dim,
            modules=modules,
            x0=x0,
            sigma0=0.3 * (ub - lb),
            budget=budget,
            lb=problem.bounds.lb,
            ub=problem.bounds.ub,
        )
        cma = c_maes.ModularCMAES(c_maes.Parameters(settings))
        cma.run(problem)

    run_gnbg_benchmark(logic, "Mirrored-CMA-ES", budget_mul, debug)


@time_iterations()
def mega_cmaes_gnbg(budget_mul: int = 2000, debug: int = 0):
    modules = c_maes.parameters.Modules()
    modules.mirrored = c_maes.options.Mirror.PAIRWISE
    modules.active = True
    modules.restart_strategy = c_maes.options.RestartStrategy.BIPOP
    modules.sampler = c_maes.options.BaseSampler.HALTON
    modules.orthogonal = True

    def logic(problem, dim, budget):
        lb, ub = problem.bounds.lb[0], problem.bounds.ub[0]
        x0 = np.random.uniform(lb, ub, size=dim)
        settings = c_maes.parameters.Settings(
            dim=dim,
            modules=modules,
            x0=x0,
            sigma0=0.3 * (ub - lb),
            budget=budget,
            lb=problem.bounds.lb,
            ub=problem.bounds.ub,
        )
        cma = c_maes.ModularCMAES(c_maes.Parameters(settings))
        cma.run(problem)

    run_gnbg_benchmark(logic, "Mega-CMA-ES", budget_mul, debug)


if __name__ == "__main__":
    cmaes_gnbg(20000, debug=10)
    bipop_cmaes_gnbg(20000, debug=10)
    ipop_cmaes_gnbg(20000, debug=10)
    mirrored_cmaes_gnbg(20000, debug=10)
    mega_cmaes_gnbg(20000, debug=10)

import argparse
import json
import ray
import os

import iohgnbg
import numpy as np
from ioh import logger
from modcma import c_maes

from benchmarks.gnbg_speed_test import time_iterations
from misc import OverBudgetException, aoc_logger, correct_aoc


@ray.remote
def _evaluate_single_problem(optimizer_fn, problem_index, budget_mul):
    """Evaluates a single problem remotely."""
    problem = iohgnbg.get_problems(
        problem_indices=[problem_index],
        instances_folder="benchmarks/gnbg/official",
    )[0]

    l2 = aoc_logger(budget_mul, upper=1e2, triggers=[logger.trigger.ALWAYS])
    problem.attach_logger(l2)
    
    dim = problem.meta_data.n_variables
    budget = dim * budget_mul

    try:
        optimizer_fn(problem, dim, budget)
    except OverBudgetException:
        pass

    auc = correct_aoc(problem, l2, budget)
    debug_info = {
        "Num evaluations": problem.state.evaluations,
        "Final target found": problem.state.final_target_found,
        "Current best": problem.state.current_best.y,
        "Problem optimum": problem.optimum.y,
        "AUC": auc,
    }
    
    return problem.meta_data.name, auc, debug_info

def run_gnbg_benchmark(optimizer_fn, name: str, budget_mul: int = 2000, debug: int = 0):
    """Generic runner for GNBG problems using Ray."""
    # Distribute the 24 problems across Ray workers
    futures = [
        _evaluate_single_problem.remote(optimizer_fn, i, budget_mul) 
        for i in range(1, 25)
    ]
    results = ray.get(futures)

    aucs = []
    debug_dict = {}
    for prob_name, auc, d_info in results:
        aucs.append(auc)
        debug_dict[prob_name] = d_info

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override number of parallel workers.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20000,
        help="Override computation budget for function evaluations.",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=0,
        help="Override log level with one of the [0, 10, 20]. 0 - no debug logs, 10 - logs top 3 hardest functions, 20 - logs all functions.",
    )
    return parser

if __name__ == "__main__":

    unlimited_budget = 10_000_000

    args = build_arg_parser().parse_args()
    if args.workers is not None:
        workers = max(1, args.workers)
    else:
        workers = 1
    budget = unlimited_budget if args.budget == -1 else args.budget
    log_level = args.log_level
    if log_level not in [0, 10, 20]:
        print("Unrecognized log level. Defaulting to 0")
        log_level = 0

    print(
    json.dumps(
        {
            "num_workers": workers,
            "computation_budget": budget,
            "log_level": log_level,
        },
        indent=2,
    )
    )

    if not ray.is_initialized():
        try:
            scratch_path = os.environ["SCRATCH"]
            ray_tmp_path = "ray_temp"
            ray.init(_temp_dir=os.path.join(scratch_path, ray_tmp_path), num_cpus=workers, ignore_reinit_error=True)
        except KeyError:
            print("Path to $SCRATCH location was not found, initializing ray in cwd")
            ray.init(num_cpus=workers, ignore_reinit_error=True)

    cmaes_gnbg(budget, debug=log_level)

    ray.shutdown()

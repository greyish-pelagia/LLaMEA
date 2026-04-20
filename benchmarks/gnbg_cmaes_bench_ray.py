import argparse
import json
import jsonlines
import ray
import os

import iohgnbg
import numpy as np
from ioh import logger
from modcma import c_maes
from pathlib import Path
from datetime import datetime as dt

from benchmarks.gnbg_speed_test import time_iterations
from misc import OverBudgetException, ThresholdReachedException, aoc_logger, correct_aoc, mae_logger, correct_mae


@ray.remote
def _evaluate_single_problem(optimizer_fn, problem_index, budget_mul):
    """Evaluates a single problem remotely."""
    problem = iohgnbg.get_problems(
        problem_indices=[problem_index],
        instances_folder="benchmarks/gnbg/official",
    )[0]

    dim = problem.meta_data.n_variables
    if problem_index >= 16:
        budget = dim * budget_mul * 2
    else:
        budget = dim * budget_mul
    budget_to_thresh = None

    l2 = aoc_logger(budget_mul, upper=1e2, triggers=[logger.trigger.ALWAYS])
    l3 = mae_logger(budget_mul, triggers=[logger.trigger.ALWAYS], stop_on_threshold=True)
    problem.attach_logger(l2)
    problem.attach_logger(l3)

    try:
        optimizer_fn(problem, dim, budget)
    except OverBudgetException:
        pass
    except ThresholdReachedException:
        budget_to_thresh = problem.state.evaluations

    auc = correct_aoc(problem, l2, budget)
    mae = correct_mae(problem, l3, budget_mul)
    final_error = float(abs(problem.state.current_best_internal.y - problem.optimum.y))
    debug_info = {
        "Num evaluations": problem.state.evaluations,
        "Final target found": True if budget_to_thresh is not None else problem.state.final_target_found,
        "Budget to thresh": budget_to_thresh,
        "Current best": problem.state.current_best.y,
        "Problem optimum": problem.optimum.y,
        "AUC": auc,
        "MAE": mae,
        "Error": final_error
    }

    l2.reset(problem)
    l3.reset(problem)
    problem.reset()
    
    return problem.meta_data.name, auc, debug_info

@ray.remote(num_cpus=0)
def run_gnbg_benchmark_task(optimizer_fn, name: str, budget_mul: int = 2000, debug: int = 0, iteration: int = 0):
    """Generic runner for GNBG problems using Ray."""
    # Distribute the 24 problems across Ray workers
    futures = [
        _evaluate_single_problem.remote(optimizer_fn, i, budget_mul) 
        for i in range(1, 25)
    ]
    results = ray.get(futures)

    aucs = []
    aes = []
    debug_dict = {}
    for prob_name, auc, d_info in results:
        aucs.append(d_info["AUC"])
        debug_dict[prob_name] = d_info
        aes.append(d_info["Error"])

    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)
    ae_mean = np.mean(aes)
    ae_std = np.std(aes)

    if debug == 10:
        top_3 = dict(sorted(debug_dict.items(), key=lambda item: item[1]["AUC"])[:3])
        print(json.dumps(top_3, indent=4))
    elif debug == 20:
        print(json.dumps(debug_dict, indent=4))

    return {
        "timestamp": dt.now().isoformat(),
        "name": name,
        "iter": iteration,
        "Average Error:": ae_mean,
        "AE std": ae_std,
        "Average AOCC": auc_mean,
        "AUCC std": auc_std,
        "Debug dict": debug_dict,

    }

def run_whole_benchmark(optimizer_fn, name: str, budget_mul: int = 2000, debug: int = 0, reps: int = 1, out_path: str = None):
    futures = [
        run_gnbg_benchmark_task.remote(optimizer_fn, name, budget_mul, debug, i)
        for i in range(reps)
    ]

    results = ray.get(futures)

    if out_path:
        for result in results:
            append_jsonl(out_path, result)

def append_jsonl(path: str | Path, payload: dict):
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(file, 'a') as f:
        f.write(payload)


@time_iterations()
def cmaes_gnbg(budget_mul: int = 2000, debug: int = 0, reps: int = 1, out_path: str = None):
    def cmaes_logic(problem, dim, budget):
        lb, ub = problem.bounds.lb[0], problem.bounds.ub[0]
        x0 = np.random.uniform(lb, ub, size=dim)
        sigma0 = 0.3 * (ub - lb)
        c_maes.fmin(problem, x0, sigma0, budget)

    run_whole_benchmark(cmaes_logic, "CMA-ES", budget_mul, debug, reps=reps, out_path=out_path)

@time_iterations()
def test_algo1(budget_mul: int = 2000, debug: int = 0, reps: int = 1, out_path: str = None):
    from benchmarks.best_algos_totest.AdaptiveMirrorAntiRank1CMARefinedV5_TrustRegionNoiseRobust import AdaptiveMirrorAntiRank1CMARefinedV5_TrustRegionNoiseRobust

    def logic(problem, dim, budget):
        alg = AdaptiveMirrorAntiRank1CMARefinedV5_TrustRegionNoiseRobust(budget=budget, dim=dim)
        alg(problem)

    run_whole_benchmark(logic, "AdaptiveMirrorAntiRank1CMARefinedV5_TrustRegionNoiseRobust", budget_mul, debug, reps=reps, out_path=out_path)

@time_iterations()
def test_algo2(budget_mul: int = 2000, debug: int = 0, reps: int = 1, out_path: str = None):
    from benchmarks.best_algos_totest.AdaptiveTrustRegionMixtureRestartCMA import AdaptiveTrustRegionMixtureRestartCMA

    def logic(problem, dim, budget):
        alg = AdaptiveTrustRegionMixtureRestartCMA(budget=budget, dim=dim)
        alg(problem)

    run_whole_benchmark(logic, "AdaptiveTrustRegionMixtureRestartCMA", budget_mul, debug, reps=reps, out_path=out_path)

@time_iterations()
def test_algo3(budget_mul: int = 2000, debug: int = 0, reps: int = 1, out_path: str = None):
    from benchmarks.best_algos_totest.candidate import Algorithm

    def logic(problem, dim, budget):
        alg = Algorithm(budget=budget, dim=dim)
        alg(problem)

    run_whole_benchmark(logic, "CodexAlgorithm", budget_mul, debug, reps=reps, out_path=out_path)


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
        "--reps",
        type=int,
        default=1,
        help="Override independent repetition count for function evaluations.",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=0,
        help="Override log level with one of the [0, 10, 20]. 0 - no debug logs, 10 - logs top 3 hardest functions, 20 - logs all functions.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to save the results in.",
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
    reps = args.reps
    log_level = args.log_level
    if log_level not in [0, 10, 20]:
        print("Unrecognized log level. Defaulting to 0")
        log_level = 0

    out_path = args.out

    print(
        json.dumps(
            {
                "num_workers": workers,
                "computation_budget": budget,
                "log_level": log_level,
                "Log output": out_path
            },
            indent=2,
        )
    )

    if not ray.is_initialized():
        try:
            scratch_path = os.environ["SCRATCH"]
            ray_tmp_path = "ray_temp"
            ray.init(_temp_dir=os.path.join(scratch_path, ray_tmp_path), num_cpus=workers)
        except KeyError:
            print("Path to $SCRATCH location was not found, initializing ray in cwd")
            ray.init(num_cpus=workers)

    test_algo3(budget, debug=log_level, reps=reps, out_path=out_path)

    ray.shutdown()
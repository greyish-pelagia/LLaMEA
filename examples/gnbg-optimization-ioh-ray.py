from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import iohgnbg
import numpy as np
import ray
from dotenv import load_dotenv
from ioh import logger

from llamea import LLaMEA, OpenRouter_LLM
from llamea.utils import clean_local_namespace, prepare_namespace
from misc import OverBudgetException, aoc_logger, correct_aoc

PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "problem_ids": [1, 2],
        "reps": 1,
        "budget_scale": 0.02,
        "parallel_workers": 1,
    },
    "search": {
        "problem_ids": list(range(1, 10)),
        "reps": 1,
        "budget_scale": 0.1,
        "parallel_workers": 6,
    },
    "hard": {
        "problem_ids": [9, 13, 3, 10, 14],
        "reps": 2,
        "budget_scale": 20.0,
        "parallel_workers": 8,
    },
    "timing": {
        "problem_ids": list(range(1, 25)),
        "reps": 3,
        "budget_scale": 20.0,
        "parallel_workers": 1,
    },
    "final": {
        "problem_ids": list(range(1, 25)),
        "reps": 31,
        "budget_scale": 20.0,
        "parallel_workers": max(1, min(8, os.cpu_count() or 1)),
    },
}

GNBG_INSTANCES_FOLDER = "benchmarks/gnbg/official"
GNBG_BASE_BUDGET = 2000


def _safe_set_failure(solution, message: str, error: Exception | None = None):
    if error is None:
        solution.set_scores(float("-inf"), message)
        return
    detail = f"{type(error).__name__}: {error}"
    solution.set_scores(float("-inf"), f"{message} {detail}")


def _validate_generated_code(code: str, algorithm_name: str, explogger=None):
    possible_issue = None
    local_ns: dict[str, Any] = {}

    try:
        global_ns, possible_issue = prepare_namespace(
            code, allowed=["numpy"], logger=explogger
        )
        exec(code, global_ns, local_ns)
        local_ns = clean_local_namespace(local_ns, global_ns)
    except Exception as e:
        message = "Generated code failed to compile or initialize."
        if possible_issue:
            message += f" {possible_issue}."
        return None, message, e

    if algorithm_name not in local_ns:
        return None, f"Generated code did not define `{algorithm_name}`.", None

    return local_ns[algorithm_name], None, None


def _normalize_problems(problems) -> list[Any]:
    if isinstance(problems, list):
        return problems
    if isinstance(problems, tuple):
        return list(problems)
    try:
        return list(problems)
    except TypeError:
        return [problems]


def _load_ioh_problem(fid: int):
    last_error: Exception | None = None

    # Prefer the explicit single-index form if the library accepts it.
    for problem_indices, selector in (
        ([fid], lambda items: items[0]),
        ((fid,), lambda items: items[0]),
        (24, lambda items: items[fid - 1]),
    ):
        try:
            problems = _normalize_problems(
                iohgnbg.get_problems(
                    problem_indices=problem_indices,
                    instances_folder=GNBG_INSTANCES_FOLDER,
                )
            )
            if not problems:
                continue
            return selector(problems)
        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"Could not load IOH GNBG problem f{fid} from {GNBG_INSTANCES_FOLDER}."
        + (" " if last_error is not None else "")
        + (
            f"Last error: {type(last_error).__name__}: {last_error}"
            if last_error
            else ""
        )
    )


def _problem_dim(problem) -> int:
    meta_data = getattr(problem, "meta_data", None)
    if meta_data is not None and hasattr(meta_data, "n_variables"):
        return int(meta_data.n_variables)
    if hasattr(problem, "dim"):
        return int(problem.dim)
    raise AttributeError(
        "Could not determine problem dimensionality from the IOH problem object."
    )


class _BoundsView:
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub


def _problem_bounds(problem, dim: int) -> tuple[np.ndarray, np.ndarray]:
    bounds = getattr(problem, "bounds", None)
    if bounds is not None and hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        return np.asarray(bounds.lb, dtype=float), np.asarray(bounds.ub, dtype=float)

    if hasattr(problem, "lower") and hasattr(problem, "upper"):
        return np.asarray(problem.lower, dtype=float), np.asarray(
            problem.upper, dtype=float
        )

    # Conservative fallback matching the built-in example prompt used by LLaMEA.
    lower = np.full(dim, -5.0, dtype=float)
    upper = np.full(dim, 5.0, dtype=float)
    return lower, upper


def _problem_optimum_y(problem) -> float:
    """
    Best-effort extraction of the problem's known optimum objective value.

    IOH problems typically expose `problem.optimum` as a RealSolution/IntegerSolution
    with a `.y` attribute. Keep a few defensive fallbacks because the exact wrapper
    around iohgnbg may differ by version.
    """
    optimum = getattr(problem, "optimum", None)
    if optimum is not None:
        for attr in ("y", "value", "objective"):
            value = getattr(optimum, attr, None)
            if value is not None:
                return float(value)

    log_info = getattr(problem, "log_info", None)
    if log_info is not None:
        objective = getattr(log_info, "objective", None)
        if objective is not None:
            return float(objective)

    raise AttributeError(
        "Could not determine the known optimum objective value from the IOH problem object."
    )


class _IOHProblemAdapter:
    """
    Makes the IOH GNBG problem look like the local GNBG interface expected by many
    generated optimizers. Exposes both:
      - func.lower / func.upper
      - func.bounds.lb / func.bounds.ub
    and enforces the scaled evaluation budget via OverBudgetException.
    """

    def __init__(self, problem, budget: int):
        self._problem = problem
        self.dim = _problem_dim(problem)
        self.budget = int(budget)
        self.lower, self.upper = _problem_bounds(problem, self.dim)
        self.bounds = _BoundsView(self.lower, self.upper)
        self.evaluations = 0
        self.best_y = float("inf")

    def __call__(self, x):
        if self.evaluations >= self.budget:
            raise OverBudgetException(
                f"Evaluation budget exceeded: {self.evaluations} >= {self.budget}"
            )

        y = np.asarray(self._problem(x))
        if y.size != 1:
            raise ValueError(f"Expected scalar objective value, got shape {y.shape}")
        y = float(y.item())

        self.evaluations += 1
        if y < self.best_y:
            self.best_y = y
        return y

    def reset(self):
        self.evaluations = 0
        self.best_y = float("inf")
        return self._problem.reset()

    def __getattr__(self, name: str):
        return getattr(self._problem, name)


@ray.remote
def _run_single_case_remote(code, algorithm_name, fid, rep, budget_scale):
    return _run_single_case(code, algorithm_name, fid, rep, budget_scale)


def _run_single_case(
    code: str,
    algorithm_name: str,
    fid: int,
    rep: int,
    budget_scale: float,
) -> dict[str, Any]:
    """
    Top-level worker function so it can run in separate processes.
    Reconstructs the algorithm from source code inside the worker.
    """
    np.random.seed(rep)

    algorithm_cls, message, error = _validate_generated_code(
        code, algorithm_name, explogger=None
    )
    if algorithm_cls is None:
        return {
            "ok": False,
            "fid": fid,
            "rep": rep,
            "error": message
            if error is None
            else f"{message} {type(error).__name__}: {error}",
        }

    try:
        problem = _load_ioh_problem(fid)
        optimum_y = _problem_optimum_y(problem)
        scaled_budget = max(1, int(GNBG_BASE_BUDGET * budget_scale))
        wrapped_problem = _IOHProblemAdapter(problem, scaled_budget)
        l2 = aoc_logger(
            scaled_budget,
            upper=1e2,
            triggers=[logger.trigger.ALWAYS],
        )
        problem.attach_logger(l2)

        algorithm = algorithm_cls(
            budget=scaled_budget,
            dim=wrapped_problem.dim,
        )

        t0 = time.perf_counter()
        try:
            algorithm(wrapped_problem)
        except OverBudgetException:
            pass
        elapsed_s = time.perf_counter() - t0

        auc = float(correct_aoc(problem, l2, scaled_budget))
        best_y = float(wrapped_problem.best_y)
        mean_error = float(abs(best_y - optimum_y))
        l2.reset(problem)
        problem.reset()

        return {
            "ok": True,
            "fid": fid,
            "rep": rep,
            "score": auc,
            "auc": auc,
            "best_y": best_y,
            "optimum_y": float(optimum_y),
            "mean_error": mean_error,
            "elapsed_s": elapsed_s,
            "budget": scaled_budget,
            "dim": wrapped_problem.dim,
            "fes": wrapped_problem.evaluations,
        }
    except Exception as e:
        return {
            "ok": False,
            "fid": fid,
            "rep": rep,
            "error": f"Runtime error on IOH GNBG f{fid}, rep {rep}. {type(e).__name__}: {e}",
        }


def _evaluate_cases(
    code: str,
    algorithm_name: str,
    problem_ids: list[int],
    reps: int,
    budget_scale: float,
    workers: int,
):
    cases = [(fid, rep) for fid in problem_ids for rep in range(reps)]

    results = []

    if workers <= 1 or len(cases) <= 1:
        results = [
            _run_single_case(code, algorithm_name, fid, rep, budget_scale)
            for fid, rep in cases
        ]
        results.sort(key=lambda item: (item["fid"], item["rep"]))
        return results

    if not ray.is_initialized():
        try:
            scratch_path = os.environ["SCRATCH"]
            ray_tmp_path = "ray_temp"
            ray.init(_temp_dir=os.path.join(scratch_path, ray_tmp_path), num_cpus=workers, ignore_reinit_error=True)
        except KeyError:
            print("Path to $SCRATCH location was not found, initializing ray in cwd")
            ray.init(num_cpus=workers, ignore_reinit_error=True)

    futures = [
        _run_single_case_remote.remote(code, algorithm_name, fid, rep, budget_scale)
        for fid, rep in cases
    ]
    results = ray.get(futures)
    results.sort(key=lambda item: (item["fid"], item["rep"]))
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_PRESETS.keys()),
        default="quick",
        help="Evaluation profile.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override number of parallel workers.",
    )
    parser.add_argument(
        "--budget-scale",
        type=float,
        default=None,
        help="Override fraction of the base FE budget to use.",
    )
    parser.add_argument(
        "--llamea-budget",
        type=int,
        default=1,
        help="Outer LLaMEA budget.",
    )
    parser.add_argument(
        "--archive-path",
        type=str,
        default=None,
        help="Path to the directory that should be taken as starting point for following LLaMEA generations",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    preset = PROFILE_PRESETS[args.profile].copy()
    if args.workers is not None:
        preset["parallel_workers"] = max(1, args.workers)
    if args.budget_scale is not None:
        preset["budget_scale"] = args.budget_scale
    archive_path = args.archive_path

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    ai_model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set in the environment or .env.")

    experiment_name = "pop1-5"
    llm = OpenRouter_LLM(api_key, ai_model)

    print(
        json.dumps(
            {
                "profile": {
                    "name": args.profile,
                    "problem_ids": preset["problem_ids"],
                    "reps": preset["reps"],
                    "budget_scale": preset["budget_scale"],
                    "parallel_workers": preset["parallel_workers"],
                    "base_budget": GNBG_BASE_BUDGET,
                    "instances_folder": GNBG_INSTANCES_FOLDER,
                },
                "llm_model": ai_model,
                "llamea_budget": args.llamea_budget,
                "experiment_name": experiment_name,
                "resume_path": archive_path
            },
            indent=2,
        )
    )

    def evaluateGNBG(solution, explogger=None):
        code = solution.code
        algorithm_name = solution.name

        algorithm_cls, message, error = _validate_generated_code(
            code, algorithm_name, explogger=explogger
        )
        if algorithm_cls is None:
            _safe_set_failure(solution, message, error)
            return solution

        # Validation succeeded; run isolated cases, reconstructing the algorithm in each worker.
        t0 = time.perf_counter()
        results = _evaluate_cases(
            code=code,
            algorithm_name=algorithm_name,
            problem_ids=list(preset["problem_ids"]),
            reps=int(preset["reps"]),
            budget_scale=float(preset["budget_scale"]),
            workers=int(preset["parallel_workers"]),
        )
        wall_time_s = time.perf_counter() - t0

        failures = [r for r in results if not r["ok"]]
        if failures:
            first = failures[0]
            _safe_set_failure(solution, first["error"])
            return solution

        run_scores = [r["score"] for r in results]
        run_mean_errors = [r["mean_error"] for r in results]
        run_details = results

        score_mean = float(np.mean(run_scores))
        score_std = float(np.std(run_scores))
        mean_error_mean = float(np.mean(run_mean_errors))
        mean_error_std = float(np.std(run_mean_errors))
        total_case_time_s = float(sum(r["elapsed_s"] for r in results))

        total_fes = int(sum(r.get("fes", 0) for r in results))

        mean_error_by_problem: dict[int, float] = {}
        for fid in sorted({r["fid"] for r in results}):
            fid_errors = [r["mean_error"] for r in results if r["fid"] == fid]
            mean_error_by_problem[fid] = float(np.mean(fid_errors))

        feedback = (
            f"The algorithm {algorithm_name} got an average GNBG AOCC score of "
            f"{score_mean:.4f} (std {score_std:.4f}), where 1.0 is the best, and "
            f"an average mean error of {mean_error_mean:.6g} (std {mean_error_std:.6g}), "
            f"where lower is better. Profile={args.profile}, cases={len(results)}, "
            f"total_fes={total_fes}, wall_time={wall_time_s:.2f}s, "
            f"summed_case_time={total_case_time_s:.2f}s."
        )

        solution.add_metadata("gnbg_backend", "iohgnbg")
        solution.add_metadata("gnbg_profile", args.profile)
        solution.add_metadata("gnbg_wall_time_s", wall_time_s)
        solution.add_metadata("gnbg_total_case_time_s", total_case_time_s)
        solution.add_metadata("gnbg_total_fes", total_fes)
        solution.add_metadata("gnbg_runs", run_details)
        solution.add_metadata("gnbg_mean_error", mean_error_mean)
        solution.add_metadata("gnbg_mean_error_std", mean_error_std)
        solution.add_metadata("gnbg_mean_error_by_problem", mean_error_by_problem)
        solution.add_metadata("gnbg_base_budget", GNBG_BASE_BUDGET)
        solution.add_metadata("gnbg_instances_folder", GNBG_INSTANCES_FOLDER)
        solution.set_scores(score_mean, feedback)
        return solution

    task_prompt = """
    The optimization algorithm will be evaluated on 24 GNBG black-box optimization
    problems. Write Python code for an optimizer with:

    - __init__(self, budget, dim)
    - __call__(self, func)

    The provided `func` is callable: `y = func(x)`.
    The optimizer must not exceed the function evaluation budget.

    The function object may expose:
    - func.dim
    - func.lower / func.upper
    - func.bounds.lb / func.bounds.ub
    - func.budget

    Prefer robust code that supports both bound interfaces when possible.

    The search space is box-constrained and may differ between problem instances.
    Your algorithm should work across different dimensions and landscapes.
    Give an excellent and novel heuristic algorithm and a one-line description of its main idea.
    """

    if archive_path:
        
        restored_llamea = LLaMEA.warm_start(archive_path)

        if restored_llamea is not None:
            additional_evals = args.llamea_budget * restored_llamea.n_offspring
            restored_llamea.budget += additional_evals
            result = restored_llamea.run()
        else:
            print(f"Failed to load archive from {archive_path}")
            exit(1)

    else:
        for experiment_i in [1]:
            es = LLaMEA(
                evaluateGNBG,
                n_parents=1,
                n_offspring=1,
                llm=llm,
                task_prompt=task_prompt,
                experiment_name=experiment_name,
                elitism=True,
                HPO=False,
                budget=args.llamea_budget,
                niching="map_elites",
            )
            result = es.run()
    print("fitness =", getattr(result, "fitness", None))
    print("feedback =", getattr(result, "feedback", None))

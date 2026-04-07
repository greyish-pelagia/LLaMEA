# This is a minimal example of how to use the LLaMEA algorithm with an LLM of choice from OpenRouter to generate optimization algorithms for the GNBG test suite.
# We have to define the following components for LLaMEA to work:
# - An evaluation function that executes the generated code and evaluates its performance.
# - A task prompt that describes the problem to be solved.
# - An LLM instance that will generate the code based on the task prompt.

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from benchmarks.gnbg.loader import make_problem
from llamea import LLaMEA, OpenRouter_LLM
from llamea.utils import clean_local_namespace, prepare_namespace
from misc import OverBudgetException

if __name__ == "__main__":
    # Execution code starts here
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    ai_model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set in the environment or .env.")
    experiment_name = "pop1-5"
    llm = OpenRouter_LLM(api_key, ai_model)

    # We define the evaluation function that executes the generated algorithm (solution.code) on the GNBG test suite.
    def evaluateGNBG(solution, explogger=None):
        code = solution.code
        algorithm_name = solution.name
        feedback = ""
        possible_issue = None
        local_ns = {}

        try:
            global_ns, possible_issue = prepare_namespace(
                code, allowed=["numpy"], logger=explogger
            )
            exec(code, global_ns, local_ns)
            local_ns = clean_local_namespace(local_ns, global_ns)
        except Exception as e:
            if possible_issue:
                feedback = f" {possible_issue}."
            solution.set_scores(float("-inf"), feedback, e)
            return solution

        if algorithm_name not in local_ns:
            solution.set_scores(
                float("-inf"),
                f"Generated code did not define `{algorithm_name}`.",
            )
            return solution

        run_scores = []
        run_details = []

        # Debug small first; switch to range(1, 25) later
        for fid in range(1, 3):
            for rep in range(1):
                np.random.seed(rep)
                problem = make_problem(fid)

                try:
                    algorithm = local_ns[algorithm_name](
                        budget=problem.budget,
                        dim=problem.dim,
                    )
                    algorithm(problem)
                except OverBudgetException:
                    pass
                except Exception as e:
                    solution.set_scores(
                        float("-inf"),
                        f"Runtime error on GNBG f{fid}, rep {rep}.",
                        e,
                    )
                    return solution

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

        feedback = (
            f"The algorithm {algorithm_name} got an average GNBG score of "
            f"{score_mean:.4f} (std {score_std:.4f}), where score = -log10(best_error)."
        )

        solution.add_metadata("gnbg_runs", run_details)
        solution.set_scores(score_mean, feedback)
        return solution

    # The task prompt describes the problem to be solved by the LLaMEA algorithm.
    task_prompt = """
    The optimization algorithm will be evaluated on 24 GNBG black-box optimization
    problems. Write Python code for an optimizer with:

    - __init__(self, budget, dim)
    - __call__(self, func)

    The provided `func` is callable: `y = func(x)`.
    The optimizer must not exceed the function evaluation budget.

    The function object may expose:
    - func.dim
    - func.lower
    - func.upper
    - func.budget

    The search space is box-constrained and may differ between problem instances.
    Your algorithm should work across different dimensions and landscapes.
    Give an excellent and novel heuristic algorithm and a one-line description of its main idea.
    """

    for experiment_i in [1]:
        # A 1+1 strategy
        es = LLaMEA(
            evaluateGNBG,
            n_parents=1,
            n_offspring=1,
            llm=llm,
            task_prompt=task_prompt,
            experiment_name=experiment_name,
            elitism=True,
            HPO=False,
            budget=1,
        )
        print(es.run())

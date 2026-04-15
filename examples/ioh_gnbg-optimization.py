# This is a minimal example of how to use the LLaMEA algorithm with the Gemini LLM to generate optimization algorithms for the GNBG II test suite.
# We have to define the following components for LLaMEA to work:
# - An evaluation function that executes the generated code and evaluates its performance.
# - A task prompt that describes the problem to be solved.
# - An LLM instance that will generate the code based on the task prompt.

import os
from pathlib import Path

import iohgnbg
import numpy as np
from dotenv import load_dotenv
from ioh import logger

from llamea import LLaMEA, OpenRouter_LLM
from llamea.utils import clean_local_namespace, prepare_namespace
from misc import OverBudgetException, aoc_logger, correct_aoc

if __name__ == "__main__":
    # Execution code starts here
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    ai_model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set in the environment or .env.")
    experiment_name = "pop1-5"
    llm = OpenRouter_LLM(api_key, ai_model)

    # We define the evaluation function that executes the generated algorithm (solution.code) on the BBOB test suite.
    # It should set the scores and feedback of the solution based on the performance metric, in this case we use mean AOCC.
    def evaluateGNBG(solution, explogger=None):
        auc_mean = 0
        auc_std = 0

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

        aucs = []

        algorithm = None
        budget = 2000
        l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])

        problems = iohgnbg.get_problems(problem_indices=24)

        for problem in problems:
            problem.attach_logger(l2)

            for rep in range(1):
                np.random.seed(rep)
                try:
                    algorithm = local_ns[algorithm_name](
                        budget=budget, dim=problem.meta_data.n_variables
                    )
                    algorithm(problem)
                except OverBudgetException:
                    pass

                auc = correct_aoc(problem, l2, budget)
                aucs.append(auc)
                l2.reset(problem)
                problem.reset()
        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)

        feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.4f} with standard deviation {auc_std:0.4f}."

        print(algorithm_name, algorithm, auc_mean, auc_std)
        solution.add_metadata("aucs", aucs)
        solution.set_scores(auc_mean, feedback)

        return solution

    # The task prompt describes the problem to be solved by the LLaMEA algorithm.
    task_prompt = """
    The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
    The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
    Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description with the main idea.
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
            budget=20,
        )
        print(es.run())

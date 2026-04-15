from __future__ import annotations

from pathlib import Path

import numpy as np

from misc import OverBudgetException

from .gnbg_cpp import GNBG

OFFICIAL_DIR = Path(__file__).resolve().parent / "official"


def load_gnbg(problem_index: int) -> GNBG:
    if not 1 <= problem_index <= 24:
        raise ValueError("problem_index must be in [1, 24]")
    return GNBG(problem_index)


class GNBGProblem:
    def __init__(self, gnbg: GNBG):
        self.gnbg = gnbg
        self.dim = int(gnbg.Dimension)
        self.lower = float(gnbg.MinCoordinate)
        self.upper = float(gnbg.MaxCoordinate)
        self.budget = int(gnbg.MaxEvals)

        self.best_y = float("inf")
        self.best_x = None

    def __call__(self, x):
        x = np.asarray(x, dtype=float).reshape(1, -1)

        if x.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {x.shape[1]}")

        if self.gnbg.FE >= self.budget:
            raise OverBudgetException("Function evaluation budget exhausted.")

        y = float(self.gnbg.fitness(x)[0])

        if y < self.best_y:
            self.best_y = y
            self.best_x = x.reshape(-1).copy()

        return y

    @property
    def optimum(self) -> float:
        return float(self.gnbg.OptimumValue)

    @property
    def acceptance_threshold(self) -> float:
        return float(self.gnbg.AcceptanceThreshold)

    @property
    def evaluations(self) -> int:
        return int(self.gnbg.FE)

    @property
    def acceptance_reach_point(self):
        return self.gnbg.AcceptanceReachPoint


def make_problem(problem_index: int) -> GNBGProblem:
    return GNBGProblem(load_gnbg(problem_index))

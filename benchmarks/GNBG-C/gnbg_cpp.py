from __future__ import annotations

from ctypes import CDLL, POINTER, c_char_p, c_double, c_int, c_void_p
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_LIB = CDLL(str(_THIS_DIR / "libgnbg.so"))

_LIB.gnbg_create.argtypes = [c_int, c_char_p]
_LIB.gnbg_create.restype = c_void_p

_LIB.gnbg_destroy.argtypes = [c_void_p]
_LIB.gnbg_destroy.restype = None

_LIB.gnbg_eval.argtypes = [c_void_p, POINTER(c_double)]
_LIB.gnbg_eval.restype = c_double

_LIB.gnbg_dimension.argtypes = [c_void_p]
_LIB.gnbg_dimension.restype = c_int

_LIB.gnbg_lower.argtypes = [c_void_p]
_LIB.gnbg_lower.restype = c_double

_LIB.gnbg_upper.argtypes = [c_void_p]
_LIB.gnbg_upper.restype = c_double

_LIB.gnbg_budget.argtypes = [c_void_p]
_LIB.gnbg_budget.restype = c_int

_LIB.gnbg_evals.argtypes = [c_void_p]
_LIB.gnbg_evals.restype = c_int

_LIB.gnbg_optimum.argtypes = [c_void_p]
_LIB.gnbg_optimum.restype = c_double

_LIB.gnbg_acceptance_threshold.argtypes = [c_void_p]
_LIB.gnbg_acceptance_threshold.restype = c_double

_LIB.gnbg_acceptance_reach_point.argtypes = [c_void_p]
_LIB.gnbg_acceptance_reach_point.restype = c_double

_LIB.gnbg_best_found.argtypes = [c_void_p]
_LIB.gnbg_best_found.restype = c_double


class GNBG:
    def __init__(self, problem_index: int):
        self._obj = _LIB.gnbg_create(problem_index, str(_THIS_DIR).encode())
        if not self._obj:
            raise RuntimeError("Failed to create GNBG instance")

    def __del__(self):
        obj = getattr(self, "_obj", None)
        if obj:
            _LIB.gnbg_destroy(obj)
            self._obj = None

    def fitness(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        out = np.empty(X.shape[0], dtype=np.float64)
        for i, row in enumerate(X):
            row = np.ascontiguousarray(row, dtype=np.float64)
            out[i] = _LIB.gnbg_eval(
                self._obj,
                row.ctypes.data_as(POINTER(c_double)),
            )
        return out

    @property
    def Dimension(self):
        return int(_LIB.gnbg_dimension(self._obj))

    @property
    def MinCoordinate(self):
        return float(_LIB.gnbg_lower(self._obj))

    @property
    def MaxCoordinate(self):
        return float(_LIB.gnbg_upper(self._obj))

    @property
    def MaxEvals(self):
        return int(_LIB.gnbg_budget(self._obj))

    @property
    def FE(self):
        return int(_LIB.gnbg_evals(self._obj))

    @property
    def OptimumValue(self):
        return float(_LIB.gnbg_optimum(self._obj))

    @property
    def AcceptanceThreshold(self):
        return float(_LIB.gnbg_acceptance_threshold(self._obj))

    @property
    def AcceptanceReachPoint(self):
        value = float(_LIB.gnbg_acceptance_reach_point(self._obj))
        return np.inf if value < 0 else value

    @property
    def BestFoundResult(self):
        return float(_LIB.gnbg_best_found(self._obj))

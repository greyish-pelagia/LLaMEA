from __future__ import annotations

import gc
import io
from contextlib import redirect_stdout
from ctypes import CDLL, POINTER, c_char_p, c_double, c_int, c_void_p
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent

PY_MAT_DIR = ROOT / "gnbg_py" / "official"
CPP_DIR = ROOT / "gnbg_c"
LIB_PATH = CPP_DIR / "libgnbg.so"

PROBLEMS = range(1, 25)
SEED = 0

# wybierz jedno:
REPEATS_PER_PROBLEM = 2000
USE_FULL_BUDGET = False


def _scalar_field(raw, name):
    return np.array([item[0] for item in raw[name].flatten()])[0, 0]


def get_py_gnbg_class():
    # oficjalny plik ma kod na top-levelu i spamuje printami przy imporcie,
    # więc tłumimy stdout tylko podczas importu
    with redirect_stdout(io.StringIO()):
        from gnbg_py.official.GNBG_instances import GNBG as PyGNBG
    return PyGNBG


def load_python_gnbg(problem_index: int):
    PyGNBG = get_py_gnbg_class()
    raw = loadmat(PY_MAT_DIR / f"f{problem_index}.mat")["GNBG"]

    return PyGNBG(
        MaxEvals=int(_scalar_field(raw, "MaxEvals")),
        AcceptanceThreshold=float(_scalar_field(raw, "AcceptanceThreshold")),
        Dimension=int(_scalar_field(raw, "Dimension")),
        CompNum=int(_scalar_field(raw, "o")),
        MinCoordinate=float(_scalar_field(raw, "MinCoordinate")),
        MaxCoordinate=float(_scalar_field(raw, "MaxCoordinate")),
        CompMinPos=np.array(raw["Component_MinimumPosition"][0, 0]),
        CompSigma=np.array(raw["ComponentSigma"][0, 0], dtype=np.float64),
        CompH=np.array(raw["Component_H"][0, 0]),
        Mu=np.array(raw["Mu"][0, 0]),
        Omega=np.array(raw["Omega"][0, 0]),
        Lambda=np.array(raw["lambda"][0, 0]),
        RotationMatrix=np.array(raw["RotationMatrix"][0, 0]),
        OptimumValue=float(_scalar_field(raw, "OptimumValue")),
        OptimumPosition=np.array(raw["OptimumPosition"][0, 0]),
    )


class CppGNBG:
    def __init__(self, problem_index: int):
        if not LIB_PATH.exists():
            raise FileNotFoundError(f"Missing shared library: {LIB_PATH}")

        self.lib = CDLL(str(LIB_PATH))

        self.lib.gnbg_create.argtypes = [c_int, c_char_p]
        self.lib.gnbg_create.restype = c_void_p

        self.lib.gnbg_destroy.argtypes = [c_void_p]
        self.lib.gnbg_destroy.restype = None

        self.lib.gnbg_eval.argtypes = [c_void_p, POINTER(c_double)]
        self.lib.gnbg_eval.restype = c_double

        self.lib.gnbg_dimension.argtypes = [c_void_p]
        self.lib.gnbg_dimension.restype = c_int

        self.lib.gnbg_lower.argtypes = [c_void_p]
        self.lib.gnbg_lower.restype = c_double

        self.lib.gnbg_upper.argtypes = [c_void_p]
        self.lib.gnbg_upper.restype = c_double

        self.lib.gnbg_budget.argtypes = [c_void_p]
        self.lib.gnbg_budget.restype = c_int

        self.obj = self.lib.gnbg_create(problem_index, str(CPP_DIR).encode())
        if not self.obj:
            raise RuntimeError("gnbg_create(...) returned null")

    def close(self):
        if getattr(self, "obj", None):
            self.lib.gnbg_destroy(self.obj)
            self.obj = None

    def __del__(self):
        self.close()

    @property
    def dim(self) -> int:
        return int(self.lib.gnbg_dimension(self.obj))

    @property
    def lower(self) -> float:
        return float(self.lib.gnbg_lower(self.obj))

    @property
    def upper(self) -> float:
        return float(self.lib.gnbg_upper(self.obj))

    @property
    def budget(self) -> int:
        return int(self.lib.gnbg_budget(self.obj))

    def eval_one(self, x: np.ndarray) -> float:
        x = np.ascontiguousarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"x must be 1D, got shape={x.shape}")
        return float(self.lib.gnbg_eval(self.obj, x.ctypes.data_as(POINTER(c_double))))


def time_python(problem_index: int, xs: np.ndarray):
    gc.collect()

    t0 = perf_counter()
    problem = load_python_gnbg(problem_index)
    t1 = perf_counter()

    y0 = float(problem.fitness(xs[0])[0])

    t2 = perf_counter()
    for x in xs:
        problem.fitness(x)
    t3 = perf_counter()

    return {
        "init_s": t1 - t0,
        "first_eval_s": t2 - t1,
        "repeat_eval_s": t3 - t2,
        "first_y": y0,
        "dim": int(problem.Dimension),
        "budget": int(problem.MaxEvals),
    }


def time_cpp(problem_index: int, xs: np.ndarray):
    gc.collect()

    t0 = perf_counter()
    problem = CppGNBG(problem_index)
    t1 = perf_counter()

    y0 = problem.eval_one(xs[0])

    t2 = perf_counter()
    for x in xs:
        problem.eval_one(x)
    t3 = perf_counter()

    result = {
        "init_s": t1 - t0,
        "first_eval_s": t2 - t1,
        "repeat_eval_s": t3 - t2,
        "first_y": y0,
        "dim": int(problem.dim),
        "budget": int(problem.budget),
    }

    problem.close()
    return result


def main():
    rows = []

    for problem_index in PROBLEMS:
        # parametry problemu bierzemy z wersji pythonowej
        py_problem = load_python_gnbg(problem_index)
        dim = int(py_problem.Dimension)
        low = float(py_problem.MinCoordinate)
        high = float(py_problem.MaxCoordinate)
        budget = int(py_problem.MaxEvals)
        del py_problem

        repeats = budget if USE_FULL_BUDGET else REPEATS_PER_PROBLEM

        rng = np.random.default_rng(SEED + problem_index)
        xs = rng.uniform(low, high, size=(repeats, dim)).astype(np.float64)

        py = time_python(problem_index, xs)
        cpp = time_cpp(problem_index, xs)

        speedup = py["repeat_eval_s"] / cpp["repeat_eval_s"]
        diff = abs(py["first_y"] - cpp["first_y"])

        rows.append(
            {
                "problem": problem_index,
                "dim": dim,
                "budget": budget,
                "repeats": repeats,
                "abs_diff_first_y": diff,
                "py_init_s": py["init_s"],
                "cpp_init_s": cpp["init_s"],
                "py_repeat_s": py["repeat_eval_s"],
                "cpp_repeat_s": cpp["repeat_eval_s"],
                "speedup": speedup,
            }
        )

        print(
            f"f{problem_index:02d} | dim={dim:2d} | repeats={repeats:7d} | "
            f"diff={diff:.3e} | py={py['repeat_eval_s']:.3f}s | "
            f"cpp={cpp['repeat_eval_s']:.3f}s | speedup={speedup:.2f}x"
        )

    print("\n=== summary ===")
    total_py = sum(r["py_repeat_s"] for r in rows)
    total_cpp = sum(r["cpp_repeat_s"] for r in rows)

    print(f"total python repeat time: {total_py:.3f}s")
    print(f"total cpp repeat time   : {total_cpp:.3f}s")
    print(f"overall speedup         : {total_py / total_cpp:.2f}x")

    fastest = max(rows, key=lambda r: r["speedup"])
    slowest = min(rows, key=lambda r: r["speedup"])

    print(f"best speedup:  f{fastest['problem']:02d} -> {fastest['speedup']:.2f}x")
    print(f"worst speedup: f{slowest['problem']:02d} -> {slowest['speedup']:.2f}x")


if __name__ == "__main__":
    main()

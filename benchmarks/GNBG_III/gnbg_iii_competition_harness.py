# gnbg_iii_competition_harness.py
import csv
import os
import time

import numpy as np
import run_algorithm_template
from scipy.io import loadmat, savemat


# --------------------------------------------------------------------------
# Utilities for loading MATLAB .mat benchmark files (GNBG struct)
# --------------------------------------------------------------------------
def _matobj_to_py(obj):
    """Recursively convert MATLAB structs/arrays from scipy.io.loadmat into plain Python types."""
    # scipy.io.matlab._mio5_params.mat_struct
    if hasattr(obj, "_fieldnames"):
        out = {}
        for name in obj._fieldnames:
            out[name] = _matobj_to_py(getattr(obj, name))
        return out
    # numpy structured scalar
    if isinstance(obj, np.void) and obj.dtype.names is not None:
        out = {}
        for name in obj.dtype.names:
            out[name] = _matobj_to_py(obj[name])
        return out
    # numpy arrays (including arrays of objects/structs)
    if isinstance(obj, np.ndarray):
        # Preserve array dimensions (important for singleton third-dim cases like RotationMatrix(:,:,1))
        if obj.dtype == object:
            return np.array(
                [_matobj_to_py(x) for x in obj.ravel()], dtype=object
            ).reshape(obj.shape)
        if obj.shape == ():
            return _matobj_to_py(obj.item())
        return obj
    # numpy scalars
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def load_gnbg_problem(mat_path):
    """Load a GNBG benchmark .mat file and return GNBG as a Python dict."""
    data = loadmat(mat_path, struct_as_record=False, squeeze_me=False)
    if "GNBG" not in data:
        raise ValueError(f"MAT-file does not contain 'GNBG' variable: {mat_path}")
    gnbg_obj = data["GNBG"]
    if isinstance(gnbg_obj, np.ndarray) and gnbg_obj.size == 1:
        gnbg_obj = gnbg_obj.reshape(-1)[0]
    GNBG = _matobj_to_py(gnbg_obj)
    if not isinstance(GNBG, dict):
        raise TypeError("Failed to convert GNBG struct to dict")

    # Normalise commonly-scalar fields so the algorithm/template can use int()/float() safely.
    keep_arrays = {
        "ComponentSigma",
        "Component_H",
        "Mu",
        "Omega",
        "lambda",
        "RotationMatrix",
        "Component_MinimumPosition",
        "OptimumPosition",
        "FEhistory",
        "H_Values",
        "SigmaPattern",
        "H_pattern",
    }
    for k, v in list(GNBG.items()):
        if k in keep_arrays:
            continue
        if isinstance(v, np.ndarray) and v.size == 1:
            GNBG[k] = v.reshape(-1)[0].item()
        elif isinstance(v, np.generic):
            GNBG[k] = v.item()

    # Ensure integer-like fields are ints (MATLAB stores them as doubles).
    for k in [
        "Dimension",
        "MaxEvals",
        "o",
        "FE",
        "FirstPoint",
        "SecondPoint",
        "DynamicPeriod",
    ]:
        if k in GNBG and GNBG[k] is not None and not (isinstance(GNBG[k], np.ndarray)):
            try:
                GNBG[k] = int(GNBG[k])
            except Exception:
                pass

    # Shape fixes for MATLAB's implicit singleton expansion:
    #   - In MATLAB, A(:,:,1) works even if A is 2-D. Ensure a 3rd dim exists in Python.
    if "RotationMatrix" in GNBG and isinstance(GNBG["RotationMatrix"], np.ndarray):
        if GNBG["RotationMatrix"].ndim == 2:
            GNBG["RotationMatrix"] = GNBG["RotationMatrix"][:, :, None]
    for k in ["Component_MinimumPosition", "Mu", "Omega", "Component_H"]:
        if k in GNBG and isinstance(GNBG[k], np.ndarray) and GNBG[k].ndim == 1:
            GNBG[k] = GNBG[k][None, :]
    for k in ["ComponentSigma", "lambda"]:
        if k in GNBG and isinstance(GNBG[k], np.ndarray) and GNBG[k].ndim == 0:
            GNBG[k] = GNBG[k].reshape(1)

    return GNBG


def dicts_to_mat_struct_array(dict_list):
    """Convert a list of dicts into a MATLAB-style struct array for savemat."""
    if len(dict_list) == 0:
        return np.empty((0, 0), dtype=[("empty", "O")])
    field_names = sorted({k for d in dict_list for k in d.keys()})
    dtype = [(k, "O") for k in field_names]
    arr = np.empty((1, len(dict_list)), dtype=dtype)
    for i, d in enumerate(dict_list):
        for k in field_names:
            arr[0, i][k] = d.get(k, np.nan)
    return arr


def dict_to_mat_struct(d):
    """Convert a dict into a 1x1 MATLAB-style struct for savemat."""
    field_names = sorted(d.keys())
    dtype = [(k, "O") for k in field_names]
    arr = np.empty((1, 1), dtype=dtype)
    for k in field_names:
        arr[0, 0][k] = d[k]
    return arr


# ==========================================================================
# GNBG-III Competition Harness
# ==========================================================================
# This script:
#   - Runs a user-supplied algorithm on all GNBG-III benchmarks
#   - Computes:
#       * Error at fixed evaluation budgets
#       * Multi-target FE-to-target for thresholds {1e-1, 1e-3, 1e-5, 1e-8}
#       * Expected Running Time (ERT) per target
#       * Basic success statistics for |f - f*| <= 1e-8
#   - Exports:
#       * GNBG_III_DetailedResults_<AlgorithmName>.mat
#       * GNBG_III_Detailed_Results_<AlgorithmName>.csv
#
# PARTICIPANTS:
#   1. Implement your algorithm in a function with signature:
#
#      [BestHistory, BestValue, BestPosition, GNBG, AcceptanceReachPoint] = ...
#          runAlgorithmTemplate(GNBG, params)
#
#      (Optionally you may return a 6th output "Extra" with custom metrics.)
#
#   2. Set AlgorithmName and AlgorithmHandle below.
#   3. Run this script. Submit the .mat and .csv files.
# ==========================================================================


def main():
    # ==================== CONFIGURATION ====================

    # Directory with GNBG-III benchmark .mat files
    benchmarkDir = "GNBG_III_Benchmarks"

    # Algorithm name (used in output filenames)
    AlgorithmName = "DErand1bin"  # <<< CHANGE THIS
    AlgorithmHandle = (
        run_algorithm_template.runAlgorithmTemplate
    )  # <<< CHANGE IF YOU USE ANOTHER FILE

    # Number of independent runs per problem (competition suggestion: 25–31)
    RunNumber = 2

    # Algorithm parameters (modify as needed for your method)
    AlgorithmParams = {}
    AlgorithmParams["PopulationSize"] = 100  # Example for population-based methods
    AlgorithmParams["F"] = 0.5  # Example (DE mutation factor)
    AlgorithmParams["Cr"] = 0.9  # Example (DE crossover rate)
    # Add any algorithm-specific fields you need, e.g.:
    # AlgorithmParams['w'] = 0.7  # inertia (for PSO)
    # AlgorithmParams['c1'] = 1.5
    # AlgorithmParams['c2'] = 1.5

    # Evaluation budgets at which error is reported
    evalPoints = [
        10000,
        50000,
        100000,
        150000,
        200000,
        250000,
        300000,
        350000,
        400000,
        450000,
        500000,
    ]
    numEvalPoints = len(evalPoints)

    # Multi-target thresholds for FE-to-target + ERT
    TargetThresholds = [1e-1, 1e-3, 1e-5, 1e-8]
    numTargets = len(TargetThresholds)

    # List of benchmark problem files
    problemFiles = [
        "F1_Unimodal_Separable.mat",
        "F2_Unimodal_FullyCoupled.mat",
        "F3_IllConditioned_Separable.mat",
        "F4_IllConditioned_Coupled.mat",
        "F5_Chain_Deceptive.mat",
        "F6_SuperLinear.mat",
        "F7_PartialSeparable.mat",
        "F8_Sparse50.mat",
        "F9_Dense90.mat",
        "F10_MixedConditioning.mat",
        "F11_Multimodal_Symmetric_Sep.mat",
        "F12_Multimodal_Symmetric_Coupled.mat",
        "F13_Multimodal_Asymmetric_Sep.mat",
        "F14_Multimodal_Asymmetric_Coupled.mat",
        "F15_HighlyMultimodal_IllConditioned.mat",
        "F16_Deceptive.mat",
        "F17_ThreeComponents_Overlapping.mat",
        "F18_ThreeComponents_MixedCond.mat",
        "F19_FiveComponents_HighAsym.mat",
        "F20_MixedBasin.mat",
        "F21_PartialSep_MultiComp.mat",
        "F22_ExtremeHybrid.mat",
        "F23_Noisy.mat",
        "F24_Dynamic.mat",
    ]

    # Which problems to test (default: all 24)
    problemsToTest = list(range(1, 25))  # 1 to 24
    # Example: problemsToTest = [1,2,3,4,5,6,11,12,13,14,15,16]

    # Pre-allocate result dict
    Results = [{} for _ in range(24)]  # we index by F1..F24 (0-indexed in Python)

    # ==================== HEADER ====================

    print("\n" + "=" * 50)
    print("  GNBG-III COMPETITION HARNESS")
    print("=" * 50)
    print(f"Algorithm: {AlgorithmName}")
    print(f"Runs per problem: {RunNumber}")
    print(f"Evaluation budget per run: {evalPoints[-1]}")
    print("Evaluation points:", end=" ")
    print(" ".join([f"{ep // 1000}K" for ep in evalPoints]))
    print("Targets for FE-to-target/ERT:", end=" ")
    for t in range(numTargets):
        print(f"{TargetThresholds[t]:.0e}", end=" ")
    print("\n" + "=" * 50 + "\n")

    totalStartTime = time.time()

    # ==================== MAIN TESTING LOOP ====================

    for p in range(len(problemsToTest)):
        probIdx = problemsToTest[p] - 1  # Convert to 0-based index

        print(f"Testing F{probIdx + 1}: {problemFiles[probIdx].replace('.mat', '')}")
        print("  Progress: ", end="")

        # Load once to get dimension and MaxEvals
        GNBG = load_gnbg_problem(os.path.join(benchmarkDir, problemFiles[probIdx]))
        MaxEvals = int(GNBG["MaxEvals"])  # competition standard: 5e5

        # Pre-allocate per-problem containers
        ErrorAtPoints = np.full((numEvalPoints, RunNumber), np.nan)
        ConvBhv = np.full((RunNumber, MaxEvals), np.nan)  # |f_best - f*| per FE
        AcceptancePoints = np.full(RunNumber, np.nan)  # FE where |f-f*|<=1e-8
        FETargets = np.full(
            (RunNumber, numTargets), np.nan
        )  # FE-to-target per threshold

        # Optional containers for algorithm-specific diagnostics
        DiversityHistory = np.full((RunNumber, 50), np.nan)  # if provided by algorithm
        ImprovementCounts = np.full(RunNumber, np.nan)
        StagnationPeriods = np.full(RunNumber, np.nan)

        problemStartTime = time.time()

        for run in range(RunNumber):
            # Simple progress indicator: dots and run number at multiples of 5
            if (run + 1) % 5 == 0:
                print(f"{run + 1} ", end="")
            else:
                print(".", end="")

            # --------------------------------------------------
            # Load fresh copy of GNBG problem
            # --------------------------------------------------
            GNBG = load_gnbg_problem(os.path.join(benchmarkDir, problemFiles[probIdx]))

            # Recommended deterministic seeding policy per (problem, run):
            np.random.seed(100000 + 1000 * (probIdx + 1) + run)

            # --------------------------------------------------
            # Call participant algorithm
            # --------------------------------------------------
            try:
                (
                    BestHistory,
                    BestValue,
                    BestPosition,
                    GNBG,
                    AcceptanceReachPoint,
                    Extra,
                ) = AlgorithmHandle(GNBG, AlgorithmParams)
            except ValueError:
                # If Extra is not returned, fall back to 5-output variant
                BestHistory, BestValue, BestPosition, GNBG, AcceptanceReachPoint = (
                    AlgorithmHandle(GNBG, AlgorithmParams)
                )
                Extra = {}

            # --------------------------------------------------
            # Normalise/validate BestHistory
            #   - Should be 1 x MaxEvals (best-so-far function value per FE)
            # --------------------------------------------------
            BestHistory = BestHistory.flatten()  # ensure 1D array

            if len(BestHistory) < MaxEvals:
                # Pad with final best value if shorter than MaxEvals
                BestHistory = np.pad(
                    BestHistory,
                    (0, MaxEvals - len(BestHistory)),
                    "constant",
                    constant_values=BestHistory[-1],
                )
            elif len(BestHistory) > MaxEvals:
                # Truncate if longer than MaxEvals
                BestHistory = BestHistory[:MaxEvals]

            # Enforce monotone non-increasing best-so-far behaviour
            for fe in range(1, MaxEvals):
                if BestHistory[fe] > BestHistory[fe - 1]:
                    BestHistory[fe] = BestHistory[fe - 1]

            # --------------------------------------------------
            # Compute error curve and FE-to-target metrics
            # --------------------------------------------------
            errorCurve = np.abs(BestHistory - GNBG["OptimumValue"])
            ConvBhv[run, :] = errorCurve

            # Error at evaluation points
            for i in range(numEvalPoints):
                idx = evalPoints[i]
                if idx <= MaxEvals:
                    ErrorAtPoints[i, run] = errorCurve[
                        idx - 1
                    ]  # -1 for 0-based indexing
                else:
                    ErrorAtPoints[i, run] = errorCurve[-1]

            # AcceptanceReachPoint: FE where |f - f*| <= 1e-8 (from algorithm)
            AcceptancePoints[run] = AcceptanceReachPoint

            # Multi-target FE-to-target
            for t in range(numTargets):
                thr = TargetThresholds[t]
                hitIdx = np.where(errorCurve <= thr)[0]
                if len(hitIdx) > 0:
                    FETargets[run, t] = hitIdx[0] + 1  # +1 for 1-based FE count
                else:
                    FETargets[run, t] = np.nan  # never reached

            # --------------------------------------------------
            # Optional diagnostics from Extra (if algorithm provides them)
            # --------------------------------------------------
            if "DiversityHistory" in Extra:
                # Expect a vector; resample / crop to 50 points if needed
                dh = Extra["DiversityHistory"].flatten()
                if len(dh) >= 50:
                    DiversityHistory[run, :] = dh[:50]
                else:
                    DiversityHistory[run, : len(dh)] = dh

            if "ImprovementCount" in Extra:
                ImprovementCounts[run] = Extra["ImprovementCount"]

            if "StagnationPeriods" in Extra:
                StagnationPeriods[run] = Extra["StagnationPeriods"]

        problemElapsedTime = time.time() - problemStartTime
        print(f" Done! ({problemElapsedTime:.1f} sec)")

        # ------------------------------------------------------
        # Aggregate per-problem statistics
        # ------------------------------------------------------
        nonInfMask = np.isfinite(AcceptancePoints)
        nonInfVal = AcceptancePoints[nonInfMask]

        # Store raw data
        Results[probIdx]["AlgorithmName"] = AlgorithmName
        Results[probIdx]["ProblemName"] = problemFiles[probIdx]
        Results[probIdx]["EvalPoints"] = evalPoints
        Results[probIdx]["ErrorAtPoints"] = ErrorAtPoints
        Results[probIdx]["ConvBhv"] = ConvBhv
        Results[probIdx]["AcceptancePoints"] = AcceptancePoints
        Results[probIdx]["Targets"] = TargetThresholds
        Results[probIdx]["FETargets"] = FETargets
        Results[probIdx]["DiversityHistory"] = DiversityHistory
        Results[probIdx]["ImprovementCounts"] = ImprovementCounts
        Results[probIdx]["StagnationPeriods"] = StagnationPeriods

        # Per-evaluation-point statistics
        Results[probIdx]["MeanErrors"] = np.mean(ErrorAtPoints, axis=1)
        Results[probIdx]["StdErrors"] = np.std(ErrorAtPoints, axis=1, ddof=1)
        Results[probIdx]["MedianErrors"] = np.median(ErrorAtPoints, axis=1)

        # Key budgets (assuming evalPoints[2]=100k, [5]=250k, end=500k)
        Results[probIdx]["MeanError100k"] = np.mean(ErrorAtPoints[2, :])
        Results[probIdx]["MeanError250k"] = np.mean(ErrorAtPoints[5, :])
        Results[probIdx]["MeanError500k"] = np.mean(ErrorAtPoints[-1, :])
        Results[probIdx]["MedianError500k"] = np.median(ErrorAtPoints[-1, :])
        Results[probIdx]["StdError500k"] = np.std(ErrorAtPoints[-1, :], ddof=1)
        Results[probIdx]["MinError500k"] = np.min(ErrorAtPoints[-1, :])
        Results[probIdx]["MaxError500k"] = np.max(ErrorAtPoints[-1, :])

        # Success rate for 1e-8 (based on AcceptancePoints)
        Results[probIdx]["SuccessRate"] = 100 * np.sum(nonInfMask) / RunNumber

        if len(nonInfVal) > 0:
            Results[probIdx]["MeanFEtoThreshold"] = np.mean(nonInfVal)
            Results[probIdx]["StdFEtoThreshold"] = np.std(nonInfVal, ddof=1)
        else:
            Results[probIdx]["MeanFEtoThreshold"] = np.inf
            Results[probIdx]["StdFEtoThreshold"] = np.nan

        # Multi-target success rates and ERT
        TargetSuccessRate = np.zeros(numTargets)
        ERT = np.zeros(numTargets)
        for t in range(numTargets):
            fe = FETargets[:, t]
            successMask = ~np.isnan(fe)
            TargetSuccessRate[t] = 100 * np.sum(successMask) / RunNumber

            if np.any(successMask):
                feFilled = fe.copy()
                feFilled[~successMask] = MaxEvals  # penalise failures by MaxEvals
                ERT[t] = np.sum(feFilled) / np.sum(successMask)
            else:
                ERT[t] = np.inf

        Results[probIdx]["TargetSuccessRate"] = TargetSuccessRate
        Results[probIdx]["ERT"] = ERT

        # Quick console summary for the strictest target (1e-8)
        print(
            f"  Summary: Err@500K = {Results[probIdx]['MeanError500k']:.2e} (±{Results[probIdx]['StdError500k']:.2e}), SR(1e-8)={Results[probIdx]['SuccessRate']:.1f}%, ",
            end="",
        )
        print("ERT(1e-8) = ", end="")
        if np.isfinite(ERT[-1]):
            print(f"{ERT[-1]:.0f} FEs")
        else:
            print("Inf")
        print()

    totalElapsedTime = time.time() - totalStartTime
    print(f"Total computation time: {totalElapsedTime / 60:.1f} minutes")

    # ==================== SAVE RESULTS (.MAT) ====================

    matFileName = f"GNBG_III_DetailedResults_{AlgorithmName}.mat"

    # Prepare data for MATLAB format
    ResultsMat = dicts_to_mat_struct_array(Results)
    AlgorithmParamsMat = dict_to_mat_struct(AlgorithmParams)
    save_dict = {
        "Results": ResultsMat,
        "AlgorithmParams": AlgorithmParamsMat,
        "evalPoints": np.array(evalPoints),
        "TargetThresholds": np.array(TargetThresholds),
        "AlgorithmName": AlgorithmName,
    }

    savemat(matFileName, save_dict)
    print(f"Detailed .MAT results saved to: {matFileName}")

    # ==================== EXPORT RESULTS TO CSV ====================

    csvFileName = f"GNBG_III_Detailed_Results_{AlgorithmName}.csv"
    print(f"\nExporting detailed per-run results to CSV: {csvFileName}")

    try:
        csvData = []

        # Header row
        header = ["Algorithm", "Problem", "Run"]
        for i in range(numEvalPoints):
            header.append(f"Error_{evalPoints[i] // 1000}K")

        header.append("Acceptance_FE_1e-8")
        header.append("Success_1e-8")

        # Multi-target FE-to-target & success flags
        for t in range(numTargets):
            header.append(f"FE_to_{TargetThresholds[t]:.0e}")
            header.append(f"Success_{TargetThresholds[t]:.0e}")

        csvData.append(header)

        # Rows: per problem, per run
        for p in range(len(problemsToTest)):
            probIdx = problemsToTest[p] - 1  # Convert to 0-based index
            for run in range(RunNumber):
                row = [AlgorithmName, f"F{probIdx + 1}", f"Run{run + 1}"]

                # Error at eval points
                for i in range(numEvalPoints):
                    row.append(Results[probIdx]["ErrorAtPoints"][i, run])

                # Acceptance FE and success flag (1e-8)
                if np.isfinite(Results[probIdx]["AcceptancePoints"][run]):
                    row.append(Results[probIdx]["AcceptancePoints"][run])
                    row.append(1)
                else:
                    row.append(np.nan)
                    row.append(0)

                # Multi-target FE-to-target
                for t in range(numTargets):
                    feVal = Results[probIdx]["FETargets"][run, t]
                    if np.isfinite(feVal):
                        row.append(feVal)
                        row.append(1)
                    else:
                        row.append(np.nan)
                        row.append(0)

                csvData.append(row)

        # Write CSV
        with open(csvFileName, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(csvData)

        print("CSV export complete.")

    except Exception as e:
        print(f"Warning: CSV export failed: {e}")

    print("\n" + "=" * 50)
    print("GNBG-III COMPETITION HARNESS FINISHED")
    print("=" * 50)


if __name__ == "__main__":
    main()

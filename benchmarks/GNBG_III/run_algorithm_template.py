# run_algorithm_template.py
import numpy as np
from scipy.spatial.distance import pdist
import fitness

def runAlgorithmTemplate(GNBG, params):
    """
    GNBG-III compatible algorithm template (DE example)

    This function is called by gnbg_iii_competition_harness.py.
    It implements a reference DE/rand/1/bin algorithm and returns:

    INPUT:
        GNBG   - benchmark struct loaded from F*_*.mat, including:
                   GNBG.Dimension
                   GNBG.MaxEvals
                   GNBG.MinCoordinate
                   GNBG.MaxCoordinate
                   GNBG.OptimumValue
        params - dict with algorithm hyperparameters, e.g.:
                   params.PopulationSize
                   params.F
                   params.Cr

    OUTPUT:
        BestHistory          - 1 x MaxEvals vector of best-so-far f(x) per FE
        BestValue            - final best objective value
        BestPosition         - final best solution vector
        GNBG                 - updated benchmark dict
        AcceptanceReachPoint - FE at which |f_best - f*| <= 1e-8 (Inf if never)
        Extra                - dict with optional diagnostics:
                                 Extra.DiversityHistory  (1 x 50)
                                 Extra.ImprovementCount
                                 Extra.StagnationPeriods

    PARTICIPANTS:
        - KEEP the function interface.
        - You may replace the internal DE logic with your algorithm
          (marked clearly below).
        - Always use [f, GNBG] = fitness(X, GNBG) for evaluations.
        - Respect GNBG.MaxEvals (do not exceed the evaluation budget).
    """

    # --------------------------------------------------------------
    #  1. Extract basic information / hyperparameters
    # --------------------------------------------------------------
    D = int(GNBG['Dimension'])
    MaxEvals = int(GNBG['MaxEvals'])

    # Lower and upper bounds (allow scalar or 1xD)
    LB = GNBG['MinCoordinate']
    UB = GNBG['MaxCoordinate']
    if np.isscalar(LB):
        LB = LB * np.ones(D)
    if np.isscalar(UB):
        UB = UB * np.ones(D)

    # Convert to 1D arrays if needed
    LB = LB.flatten()
    UB = UB.flatten()

    # Algorithm hyperparameters with sensible defaults
    NP = params.get('PopulationSize', 100)
    F = params.get('F', 0.5)
    Cr = params.get('Cr', 0.9)

    # --------------------------------------------------------------
    #  2. Initialise population
    # --------------------------------------------------------------
    # X: NP x D population matrix
    X = LB + (UB - LB) * np.random.rand(NP, D)

    # Best-history buffer: best-so-far value at each FE (1..MaxEvals)
    BestHistory = np.full(MaxEvals, np.inf)

    # Diversity sampling schedule (50 points over [1, MaxEvals])
    divSampleFEs = np.round(np.linspace(1, MaxEvals, 50)).astype(int)
    divIndex = 0
    DiversityHistory = np.full(50, np.nan)

    # Improvement / stagnation counters
    ImprovementCount = 0
    StagnationCount = 0
    currentStagnation = 0

    # Ensure GNBG.FE is initialised
    if 'FE' not in GNBG or GNBG['FE'] is None:
        GNBG['FE'] = 0
    prevFE = GNBG['FE']

    # Initial evaluation
    fitVals, GNBG = fitness.fitness(X, GNBG)   # increments GNBG.FE by NP
    BestValue = np.min(fitVals)
    bestID = np.argmin(fitVals)
    BestPosition = X[bestID, :].copy()

    # Fill BestHistory for these initial evaluations
    newFE = GNBG['FE']
    BestHistory[prevFE:newFE] = BestValue
    prevFE = newFE

    lastBest = BestValue   # for relative improvement tracking

    # --------------------------------------------------------------
    #  3. Main optimisation loop
    # --------------------------------------------------------------
    while GNBG['FE'] < MaxEvals:

        # ==========================================================
        #  BEGIN: REPLACE THIS BLOCK WITH YOUR OWN ALGORITHM
        # ==========================================================

        # ------------------------------
        # 3.1 Mutation: DE/rand/1
        # ------------------------------
        R = np.zeros((NP, 3), dtype=int)
        for i in range(NP):
            idx = np.random.permutation(NP)
            idx = idx[idx != i]     # exclude self
            R[i, :] = idx[:3]

        V = X[R[:, 0], :] + F * (X[R[:, 1], :] - X[R[:, 2], :])  # donor vectors

        # ------------------------------
        # 3.2 Crossover: binomial
        # ------------------------------
        U = X.copy()  # trial vectors start as copy of X

        # Ensure at least one dimension is taken from donor
        K = np.ravel_multi_index((np.arange(NP), np.random.randint(0, D, NP)),
                                 (NP, D))
        U.flat[K] = V.flat[K]

        # Apply binomial crossover mask
        crossMask = np.random.rand(NP, D) < Cr
        U[crossMask] = V[crossMask]

        # ------------------------------
        # 3.3 Boundary handling (clipping)
        # ------------------------------
        U = np.maximum(U, LB)
        U = np.minimum(U, UB)

        # ------------------------------
        # 3.4 Evaluate offspring
        # ------------------------------
        oldFE = GNBG['FE']
        fitOff, GNBG = fitness.fitness(U, GNBG)   # increments FE by #offspring
        newFE = GNBG['FE']

        # ------------------------------
        # 3.5 Selection
        # ------------------------------
        better = fitOff < fitVals
        X[better, :] = U[better, :]
        fitVals[better] = fitOff[better]

        # ------------------------------
        # 3.6 Update global best
        # ------------------------------
        currBest = np.min(fitVals)
        currID = np.argmin(fitVals)
        if currBest < BestValue:
            BestValue = currBest
            BestPosition = X[currID, :].copy()

        # ------------------------------
        # 3.7 Update BestHistory for each FE consumed
        # ------------------------------
        BestHistory[oldFE:newFE] = BestValue
        prevFE = newFE

        # ------------------------------
        # 3.8 Improvement & stagnation tracking
        # ------------------------------
        if lastBest != 0:
            relImprovement = (lastBest - BestValue) / abs(lastBest)
        else:
            relImprovement = 0

        if relImprovement > 0.01:   # >1% improvement
            ImprovementCount += 1
            lastBest = BestValue
            currentStagnation = 0
        else:
            currentStagnation += 1

        if currentStagnation > 100:
            StagnationCount += 1
            currentStagnation = 0

        # ------------------------------
        # 3.9 Diversity snapshots (50 points)
        # ------------------------------
        while (divIndex < len(divSampleFEs) and
               GNBG['FE'] >= divSampleFEs[divIndex]):
            # mean pairwise distance among individuals
            if NP > 1:
                DiversityHistory[divIndex] = np.mean(pdist(X))
            else:
                DiversityHistory[divIndex] = 0
            divIndex += 1

        # Optional early stopping if extremely close to optimum
        if 'OptimumValue' in GNBG:
            if abs(BestValue - GNBG['OptimumValue']) <= 1e-12:
                # Fill remaining BestHistory entries and terminate
                if newFE < MaxEvals:
                    BestHistory[newFE:MaxEvals] = BestValue
                GNBG['FE'] = MaxEvals
                break

        # ==========================================================
        #  END: YOUR ALGORITHM BLOCK
        # ==========================================================

        # Safety break (should normally be controlled by fitness/MaxEvals)
        if GNBG['FE'] >= MaxEvals:
            break

    # --------------------------------------------------------------
    #  4. Finalise BestHistory and compute AcceptanceReachPoint
    # --------------------------------------------------------------
    # Fill any remaining uninitialised entries with final BestValue
    finite_mask = np.isfinite(BestHistory)
    if np.any(finite_mask):
        lastFilled = np.where(finite_mask)[0][-1]
        if lastFilled < MaxEvals - 1:
            BestHistory[lastFilled+1:] = BestValue
    else:
        BestHistory[:] = BestValue

    # Compute FE at which |f_best - f*| <= 1e-8
    AcceptanceReachPoint = np.inf
    if 'OptimumValue' in GNBG:
        err = np.abs(BestHistory - GNBG['OptimumValue'])
        idx = np.where(err <= 1e-8)[0]
        if len(idx) > 0:
            AcceptanceReachPoint = idx[0] + 1  # +1 for 1-based FE index

    # --------------------------------------------------------------
    #  5. Extra diagnostics
    # --------------------------------------------------------------
    Extra = {}z    Extra['DiversityHistory'] = DiversityHistory
    Extra['ImprovementCount'] = ImprovementCount
    Extra['StagnationPeriods'] = StagnationCount

    return BestHistory, BestValue, BestPosition, GNBG, AcceptanceReachPoint, Extra

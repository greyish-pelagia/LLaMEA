# fitness.py
import numpy as np

def fitness(X, GNBG):
    """
    GNBG-III Fitness Function with Noise and Dynamic Support
    
    This function includes the objective function of GNBG based on 
    parameter settings defined by the user and stored in 'GNBG' structure.
    In addition, the results of the algorithms is gathered in this function
    and stored in 'GNBG' structure. 
    
    MODIFICATIONS FOR GNBG-III:
    - Support for F23 (noisy evaluation)
    - Support for F24 (dynamic landscapes)
    
    Reference: 
    D. Yazdani, M. N. Omidvar, D. Yazdani, K. Deb, and A. H. Gandomi, "GNBG: A Generalized
    and Configurable Benchmark Generator for Continuous Numerical Optimization," arXiv preprint arXiv:2312.07083, 2023.
    
    If you are using GNBG and this code in your work, you should cite the reference provided above.
    """
    SolutionNumber = X.shape[0]
    result = np.full(SolutionNumber, np.nan)
    
    # Check if dynamic shift is needed (F24)
    if 'DynamicShift' in GNBG and 'DynamicPeriod' in GNBG and GNBG['DynamicPeriod'] not in (0, 0.0):
        dyn_period = int(GNBG['DynamicPeriod'])
        if GNBG['FE'] % dyn_period == 0 and GNBG['FE'] > 0:
            # Apply dynamic shift to component positions
            shift = GNBG['DynamicShift'] * np.random.randn(int(GNBG['o']), int(GNBG['Dimension']))
            GNBG['Component_MinimumPosition'] = GNBG['Component_MinimumPosition'] + shift
            
            # Keep within bounds
            GNBG['Component_MinimumPosition'] = np.maximum(GNBG['Component_MinimumPosition'], GNBG['MinRandOptimaPos'])
            GNBG['Component_MinimumPosition'] = np.minimum(GNBG['Component_MinimumPosition'], GNBG['MaxRandOptimaPos'])
            
            # Update optimum position
            GNBG['OptimumValue'] = np.min(GNBG['ComponentSigma'])
            GlobalOptimumID = np.argmin(GNBG['ComponentSigma'])
            GNBG['OptimumPosition'] = GNBG['Component_MinimumPosition'][GlobalOptimumID, :]
            
            # Optional: Print shift notification
            # print(f'Dynamic shift applied at FE={GNBG["FE"]}')
    
    for jj in range(SolutionNumber):
        x = X[jj, :].reshape(-1, 1)  # Column vector
        f = np.full(GNBG['o'], np.nan)
        
        for k in range(GNBG['o']):
            # Transform operation
            a_term = x.T - GNBG['Component_MinimumPosition'][k, :].reshape(1, -1)
            a_term_rotated = np.dot(a_term, GNBG['RotationMatrix'][:, :, k].T)
            a = transform(a_term_rotated.flatten(), 
                         GNBG['Mu'][k, :], 
                         GNBG['Omega'][k, :])
            
            b_term = np.dot(GNBG['RotationMatrix'][:, :, k], 
                           x - GNBG['Component_MinimumPosition'][k, :].reshape(-1, 1))
            b = transform(b_term.flatten(), 
                         GNBG['Mu'][k, :], 
                         GNBG['Omega'][k, :])
            
            # Reshape for matrix multiplication
            a = a.reshape(1, -1)
            b = b.reshape(-1, 1)
            
            # Calculate component value
            f[k] = GNBG['ComponentSigma'][k] + \
                  (np.dot(np.dot(a, np.diag(GNBG['Component_H'][k, :])), b) ** GNBG['lambda'][k])
        
        result[jj] = np.min(f)
        
        # Add noise for F23 (Noisy benchmark)
        if 'NoiseLevel' in GNBG:
            result[jj] = result[jj] + GNBG['NoiseLevel'] * np.random.randn()
        
        if GNBG['FE'] > GNBG['MaxEvals']:
            return result, GNBG
        
        GNBG['FE'] = GNBG['FE'] + 1
        fe_idx = int(GNBG['FE']) - 1  # MATLAB FEhistory is 1-based; Python is 0-based
        GNBG['FEhistory'].flat[fe_idx] = result[jj]

        # Update best found result
        if GNBG['BestFoundResult'] > result[jj]:
            GNBG['BestFoundResult'] = result[jj]

        # Check acceptance threshold
        if (abs(GNBG['FEhistory'].flat[fe_idx] - GNBG['OptimumValue']) < GNBG['AcceptanceThreshold'] and
            np.isinf(GNBG['AcceptanceReachPoint'])):
            GNBG['AcceptanceReachPoint'] = GNBG['FE']

        # Record checkpoints
        if GNBG['FE'] == GNBG['FirstPoint']:
            GNBG['BestAtFirstLine'] = np.min(GNBG['FEhistory'].reshape(-1)[:int(GNBG['FE'])])
        if GNBG['FE'] == GNBG['SecondPoint']:
            GNBG['BestAtSecondLine'] = np.min(GNBG['FEhistory'].reshape(-1)[:int(GNBG['FE'])])
    
    return result, GNBG


def transform(X, Alpha, Beta):
    """
    Transform function for GNBG
    """
    Y = X.copy()
    
    # Positive values
    pos_mask = X > 0
    if np.any(pos_mask):
        Y[pos_mask] = np.log(X[pos_mask])
        Y[pos_mask] = np.exp(Y[pos_mask] + 
                            Alpha[0] * (np.sin(Beta[0] * Y[pos_mask]) + 
                                       np.sin(Beta[1] * Y[pos_mask])))
    
    # Negative values
    neg_mask = X < 0
    if np.any(neg_mask):
        Y[neg_mask] = np.log(-X[neg_mask])
        Y[neg_mask] = -np.exp(Y[neg_mask] + 
                             Alpha[1] * (np.sin(Beta[2] * Y[neg_mask]) + 
                                        np.sin(Beta[3] * Y[neg_mask])))
    
    return Y
# TSRI-CSQI: Translation, Scale, and Rotation Invariant operator merged with Conjugate Subspace Quadratic Interlacing to hybridize GECCO winner concepts with powerful exact parabolic leaps.
import numpy as np

class TSRICSQI:
    def __init__(self, budget=60000, dim=30):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        try:
             lb = func.bounds.lb[0]
             ub = func.bounds.ub[0]
        except AttributeError:
             lb, ub = -100.0, 100.0
             
        evals = [0]
        def obj(x):
            if evals[0] >= self.budget: return self.f_opt
            xc = np.clip(x, lb, ub)
            val = func(xc)
            evals[0] += 1
            if val < self.f_opt:
                self.f_opt = val
                self.x_opt = xc.copy()
            return val
            
        def line_search(p, d, step_init):
            if evals[0] >= self.budget: return p, obj(p)
            f_p = obj(p)
            
            p_plus = p + step_init * d
            f_plus = obj(p_plus)
            if evals[0] >= self.budget: return self.x_opt, self.f_opt
            
            p_minus = p - step_init * d
            f_minus = obj(p_minus)
            if evals[0] >= self.budget: return self.x_opt, self.f_opt
            
            denom = f_plus + f_minus - 2 * f_p
            best_x, best_f = p, f_p
            
            if f_plus < best_f: best_x, best_f = p_plus, f_plus
            if f_minus < best_f: best_x, best_f = p_minus, f_minus
                
            if denom > 1e-8:
                jump_len = 0.5 * step_init * (f_minus - f_plus) / denom
                jump_len = np.clip(jump_len, -15 * step_init, 15 * step_init)
                
                if abs(jump_len) > 1e-6:
                    p_jump = p + jump_len * d
                    if evals[0] < self.budget:
                        f_jump = obj(p_jump)
                        if f_jump < best_f:
                            best_x, best_f = p_jump, f_jump
                            
            return best_x, best_f

        archive = []
        restarts = 0
        
        while evals[0] < self.budget:
            if restarts == 0:
                x0 = (np.ones(self.dim) * (lb + ub)) / 2.0
            else:
                if len(archive) >= 3 and np.random.rand() < 0.8:
                    idx = np.random.choice(len(archive), 3, replace=False)
                    w1, w2 = np.random.randn(2)
                    w3 = 1.0 - w1 - w2
                    x0 = w1 * archive[idx[0]] + w2 * archive[idx[1]] + w3 * archive[idx[2]]
                    
                    w0 = np.random.normal(0, 0.1)
                    x0 += w0 * (ub - lb)
                    x0 = np.clip(x0, lb, ub)
                else:
                    x0 = np.random.uniform(lb, ub, self.dim)
                    
            step = (ub - lb) * 0.1
            U = np.eye(self.dim)
            
            for epoch in range(100):
                if evals[0] >= self.budget: break
                    
                x_start = x0.copy()
                f_start = obj(x_start)
                
                max_delta = 0.0
                dir_idx = 0
                
                for i in range(self.dim):
                    if evals[0] >= self.budget: break
                    d = U[i]
                    x_new, f_new = line_search(x0, d, step)
                    
                    delta = f_start - f_new
                    if delta > max_delta:
                        max_delta = delta
                        dir_idx = i
                        
                    x0 = x_new
                    f_start = f_new
                    
                if evals[0] >= self.budget: break
                
                new_d = x0 - x_start
                norm_d = np.linalg.norm(new_d)
                
                if norm_d > 1e-8:
                    new_d = new_d / norm_d
                    x0, f_new = line_search(x0, new_d, step)
                    
                    for i in range(dir_idx, self.dim - 1):
                        U[i] = U[i+1]
                    U[-1] = new_d
                else:
                    step *= 0.5
                    if step < 1e-5:
                        break 
                        
            archive.append(self.x_opt.copy())
            if len(archive) > 10:
                archive.pop(0)
            restarts += 1
            
        return self.f_opt, self.x_opt

import numpy as np

class AdaptiveTrustRegionMixtureRestartCMA:
    """
    Refined version of the selected algorithm:
      - Standard-ish CMA-ES core with trust-region shrinking and mirrored sampling.
      - Archive-driven Gaussian-mixture basin hopping, but refined:
          (a) Build a small mixture surrogate from archive top points using
              a fitness-temperature and distance-regularization.
          (b) Create a "mixture candidate mean" and *blend it into CMA*
              recombination gradually (continuous exploitation), not only on restart.
          (c) When stagnation is detected, do a stronger restart by resetting
              mean using archive mixture and keeping covariance partially informed.
      - Robust bounded handling for both {lower/upper} and {bounds.lb/ub}.
      - Hard budget cap on number of func(x) calls.
    """

    def __init__(self, budget, dim, seed=None):
        self.budget = int(budget)
        self.dim = int(dim)
        self.rng = np.random.default_rng(seed)
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        n = int(getattr(func, "dim", self.dim))
        lb, ub = self._get_bounds(func, n)
        lb = lb.astype(float)
        ub = ub.astype(float)

        def project(x):
            x = np.asarray(x, dtype=float)
            return np.minimum(ub, np.maximum(lb, x))

        max_evals = self.budget
        evals = 0

        def eval_once(x):
            nonlocal evals
            if evals >= max_evals:
                return float("inf"), project(x)
            x = project(x)
            y = float(func(x))
            evals += 1
            return y, x

        # ---- initialize mean and scale ----
        center = 0.5 * (lb + ub)
        if np.any(~np.isfinite(center)):
            finite_lb = np.where(np.isfinite(lb), lb, -1.0)
            finite_ub = np.where(np.isfinite(ub), ub, 1.0)
            center = self.rng.uniform(finite_lb, finite_ub)

        mean = project(center)

        span = ub - lb
        span = np.where(span > 0, span, 1.0)

        # ---- population sizes ----
        lam = int(4 + 3 * np.log(n + 1))
        lam = max(10, lam)
        lam = min(lam, 80)
        mu = lam // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)

        # ---- CMA parameters (conservative) ----
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / max(1, (n + 1))) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

        sigma = 0.35 * float(np.mean(span))
        sigma_min = max(1e-14, 1e-12 * float(np.mean(span))) if np.mean(span) > 0 else 1e-12

        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)

        pc = np.zeros(n)
        ps = np.zeros(n)

        tr_init = 1.0
        tr_final = 0.02

        # ---- archive for mixture ----
        # Keep more points than before to build a stable mixture, but not too many.
        archive_cap = max(30, min(360, self.budget // 35))
        archive_X = []
        archive_F = []

        # ---- best-so-far ----
        best_f = float("inf")
        best_x = mean.copy()

        # evaluate initial mean
        f0, x0 = eval_once(mean)
        if f0 < best_f:
            best_f, best_x = f0, x0.copy()
            self.f_opt, self.x_opt = best_f, best_x.copy()
        archive_X.append(best_x.copy())
        archive_F.append(best_f)

        # stagnation tracker
        window = max(12, int(lam * 3.5))
        best_hist = [best_f]

        last_eig_gen = -1
        eigeneval_every = max(14, lam // 2)

        def _archive_arrays():
            if len(archive_X) == 0:
                return None, None
            aX = np.asarray(archive_X, dtype=float)
            aF = np.asarray(archive_F, dtype=float)
            return aX, aF

        def mixture_centroid(target_center, k=10):
            """
            Refined mixture centroid:
              - choose k best archive points
              - fitness weights via exp(-(F-fmin)/T) with adaptive T
              - also include repulsion/regularization via distance to target_center
              - return centroid of weighted component means (here we use actual points)
            """
            aX, aF = _archive_arrays()
            if aX is None:
                return target_center.copy()

            idx = np.argsort(aF)
            idx = idx[: min(k, len(idx))]
            Xk = aX[idx]
            Fk = aF[idx]

            fmin = float(Fk[0])
            fmax = float(np.max(Fk))
            spread = fmax - fmin

            # Adaptive temperature: use both spread and size, prevent collapse.
            # Also depend on current sigma: when sigma small, allow sharper basin selection.
            cur_sig = max(sigma, sigma_min)
            # typical spatial scale:
            dist_scale = float(np.mean(span) + 1e-12)
            T = (spread / max(1.0, len(Fk))) * (0.7 + 0.6 * (cur_sig / (dist_scale + 1e-12)))
            if not np.isfinite(T) or T <= 1e-16:
                T = max(1e-12, spread / max(1.0, len(Fk)) + 1e-12)

            d2 = np.sum((Xk - target_center[None, :]) ** 2, axis=1)
            # distance regularization:
            # use current trust-region-ish scale (sigma times mean C scaling)
            # so that centroid prefers points close to target when exploitation is needed.
            spatial_scale = float((cur_sig ** 2) * (0.5 + 1.5 * float(tr_final / tr_init) + 1e-12))
            spatial_scale = max(spatial_scale, (0.15 * np.mean(span)) ** 2 + 1e-12)

            w_fit = np.exp(-(Fk - fmin) / T)
            # mild distance penalty (keep attraction but avoid erratic jumps)
            w_dist = np.exp(-0.15 * d2 / spatial_scale)
            w = w_fit * w_dist

            s = float(np.sum(w))
            if not np.isfinite(s) or s <= 0:
                return Xk[0].copy()

            w /= s
            m = np.sum(Xk * w[:, None], axis=0)
            return project(m)

        def sample_population():
            # trust-region shrink based on eval progress
            frac = evals / max(1, max_evals)
            tr = tr_init * (tr_final / tr_init) ** frac
            sigma_eff = max(sigma * tr, sigma_min)

            Z = np.zeros((lam, n), dtype=float)
            half = lam // 2
            if half > 0:
                z = self.rng.normal(0.0, 1.0, size=(half, n))
                Z[0:2 * half:2] = z
                Z[1:2 * half:2] = -z
            if lam % 2 == 1:
                Z[-1] = self.rng.normal(0.0, 1.0, size=n)

            # Map to x = mean + sigma_eff * (B * (D*z))
            t = (B @ (D[:, None] * Z.T)).T
            X = mean[None, :] + sigma_eff * t
            return X, Z

        # mixture injection state
        mix_k = 12
        mix_blend = 0.0  # increases during exploitation/stagnation

        def restart_strong():
            nonlocal mean, sigma, C, B, D, pc, ps, best_hist, mix_blend
            # stronger reset centered at mixture basin and best_x anchor
            mix = mixture_centroid(best_x, k=min(16, mix_k + 6))
            # convex combo leaning to mixture when sigma is small / stagnation suspected
            srel = float(sigma / (sigma_min + 1e-30))
            w_mix = 0.45 + 0.35 * (1.0 / (1.0 + srel))  # more mixture if sigma collapsed
            mean = project((1.0 - w_mix) * best_x + w_mix * mix)

            # keep covariance partially but broaden to escape narrow basins
            sigma = max(0.55 * float(np.mean(span)), sigma_min * 20.0)

            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            best_hist = [best_f]
            mix_blend = 0.0

        gen = 0
        while evals + lam <= max_evals:
            gen += 1

            Xcand, Z = sample_population()
            F = np.empty(lam, dtype=float)

            # Evaluate
            for i in range(lam):
                F[i], Xcand[i] = eval_once(Xcand[i])

            idx = np.argsort(F)
            F = F[idx]
            Xcand = Xcand[idx]
            Z = Z[idx]

            # best update
            if F[0] < best_f:
                best_f = float(F[0])
                best_x = Xcand[0].copy()
                self.f_opt, self.x_opt = best_f, best_x.copy()

            # archive update: keep current generation top candidate and some diversity
            archive_X.append(best_x.copy())
            archive_F.append(best_f)
            if len(archive_X) > archive_cap:
                aX, aF = _archive_arrays()
                order = np.argsort(aF)
                keep = archive_cap // 2
                top = order[:keep]
                if keep < archive_cap:
                    rem = order[keep:]
                    if len(rem) > 0:
                        add = self.rng.choice(rem, size=archive_cap - keep, replace=False)
                        top = np.concatenate([top, add])
                archive_X = [archive_X[i] for i in top]
                archive_F = [archive_F[i] for i in top]

            # stagnation analysis
            best_hist.append(best_f)
            if len(best_hist) > window:
                best_hist = best_hist[-window:]
                start = float(best_hist[0])
                end = float(best_hist[-1])
                rel_impr = (start - end) / max(1e-12, abs(start))
                sigma_too_small = sigma <= 3.5 * sigma_min
                sigma_too_big = sigma >= 0.9 * float(np.mean(span)) * 2.5
                spread_fit = float(np.std(best_hist)) + 1e-12
                plateau = (start - end) <= 1e-6 * spread_fit

                if rel_impr <= 1e-10 and (sigma_too_small or sigma_too_big or plateau):
                    restart_strong()

            # Eigen refresh
            if gen - last_eig_gen >= eigeneval_every:
                C = 0.5 * (C + C.T)
                try:
                    evals_c, evecs = np.linalg.eigh(C)
                    evals_c = np.maximum(evals_c, 1e-30)
                    D = np.sqrt(evals_c)
                    B = evecs
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                last_eig_gen = gen

            # ---- refined mixture injection into CMA recombination ----
            # Increase blend when stagnation likely: small sigma + no improvement
            # (kept soft to avoid derailing CMA).
            if len(best_hist) >= max(6, window // 3):
                recent = np.asarray(best_hist[-max(6, window // 3):], dtype=float)
                recent_std = float(np.std(recent)) + 1e-12
                progress = float((recent[0] - recent[-1]) / max(1e-12, abs(recent[0])))
                # if progress is tiny and sigma is small -> stronger injection
                stagn = max(0.0, 1.0 - 10.0 * max(0.0, progress))
                srel = float(sigma / (sigma_min + 1e-30))
                sigma_small = max(0.0, 1.0 - srel / 2.0)  # increases as srel->0
                mix_blend = 0.70 * max(0.0, stagn) + 0.55 * sigma_small
                mix_blend = float(np.clip(mix_blend, 0.0, 0.55))
            else:
                mix_blend = 0.0

            # Recombine CMA mean using top mu
            old_mean = mean.copy()
            mean_cma = np.sum(Xcand[:mu].T * weights[:mu], axis=1)

            # Mixture candidate centered around best_x/mean to exploit basins
            mix_center = mixture_centroid(best_x, k=min(mix_k, max(6, mu + 2)))
            # Gradual blending: mixture pulls mean towards basin but CMA maintains local structure.
            mean = project((1.0 - mix_blend) * mean_cma + mix_blend * mix_center)

            # ---- CMA path updates ----
            y = (mean - old_mean) / max(sigma, 1e-300)

            By = B.T @ y
            invsqrtC_y = B @ (By / np.maximum(D, 1e-12))

            ps = (1.0 - c_sigma) * ps + np.sqrt(c_sigma * (2.0 - c_sigma) * mueff) * invsqrtC_y

            denom = np.sqrt(max(1e-12, 1.0 - (1.0 - c_sigma) ** (2.0 * gen)))
            lhs = np.linalg.norm(ps) / denom
            rhs = (1.4 + 2.0 / (n + 1.0)) * chi_n
            h_sigma = 1.0 if lhs < rhs else 0.0

            pc = (1.0 - c_c) * pc + h_sigma * np.sqrt(c_c * (2.0 - c_c)) * y

            # Covariance update (rank-mu + rank-1)
            artmp = (Xcand[:mu] - old_mean[None, :]) / max(sigma, 1e-300)
            weighted_artmp = artmp.T * weights[:mu]
            deltaC_mu = weighted_artmp @ artmp

            I = np.eye(n)
            C = (
                (1.0 - c1 - c_mu) * C
                + c1 * (np.outer(pc, pc) + (1.0 - h_sigma) * c_c * (2.0 - c_c) * I)
                + c_mu * deltaC_mu
            )

            # Step-size update
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1.0))

            # sigma cap to avoid runaway; also lower cap
            sig_cap = 3.0 * float(np.mean(span)) + 1e-12
            sigma = float(np.clip(sigma, sigma_min, sig_cap))

        # ---- final budget-safe local mirrored search around best ----
        while evals < max_evals:
            rem = max_evals - evals
            frac = (max_evals - rem) / max(1, max_evals)
            tr = tr_init * (tr_final / tr_init) ** frac
            local_sigma = max(0.6 * sigma * tr, sigma_min)

            z = self.rng.normal(0.0, 1.0, size=n)
            x1 = project(best_x + local_sigma * z)
            f1, x1 = eval_once(x1)
            if f1 < self.f_opt:
                self.f_opt = f1
                best_x = x1.copy()

            if evals >= max_evals:
                break

            x2 = project(best_x - local_sigma * z)
            f2, x2 = eval_once(x2)
            if f2 < self.f_opt:
                self.f_opt = f2
                best_x = x2.copy()

        self.x_opt = best_x
        return self.f_opt, self.x_opt

    # ------------------------- bounds handling -------------------------
    def _get_bounds(self, func, n):
        if hasattr(func, "bounds") and func.bounds is not None:
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
            else:
                lb, ub = self._infer_bounds_from_func_attrs(func, n)
        else:
            lb, ub = self._infer_bounds_from_func_attrs(func, n)

        lb = self._ensure_vector(lb, n)
        ub = self._ensure_vector(ub, n)

        if np.any(~np.isfinite(lb)) or np.any(~np.isfinite(ub)):
            lb = -5.0 * np.ones(n)
            ub = 5.0 * np.ones(n)

        mid = 0.5 * (lb + ub)
        width = np.abs(ub - lb)
        bad = (ub <= lb) | (width <= 0) | ~np.isfinite(width)
        if np.any(bad):
            w = np.maximum(1.0, 0.1 * np.abs(mid) + 1.0)
            lb = np.where(bad, mid - w, lb)
            ub = np.where(bad, mid + w, ub)

        return lb, ub

    def _infer_bounds_from_func_attrs(self, func, n):
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            lb = -5.0 * np.ones(n)
            ub = 5.0 * np.ones(n)
        return lb, ub

    def _ensure_vector(self, a, n):
        a = np.asarray(a, dtype=float)
        if a.ndim == 0:
            return np.full(n, float(a))
        if a.size == n:
            return a.reshape(n)
        if a.size == 1:
            return np.full(n, float(a.reshape(-1)[0]))
        out = np.empty(n, dtype=float)
        flat = a.reshape(-1)
        m = min(n, flat.size)
        out[:m] = flat[:m]
        fill = flat[-1] if flat.size > 0 else 0.0
        out[m:] = fill
        out[~np.isfinite(out)] = 0.0
        return out
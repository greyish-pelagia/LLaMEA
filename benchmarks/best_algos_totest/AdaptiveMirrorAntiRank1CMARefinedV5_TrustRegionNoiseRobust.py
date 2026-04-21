import numpy as np

class AdaptiveMirrorAntiRank1CMARefinedV5_TrustRegionNoiseRobust:
    """
    Refinement V5 (focused on improving robustness/efficiency):
      - Trust-region rank-one layer:
          * Maintain a centroid-centered trust radius Tr (in decision space units).
          * Proposals are clipped/warped so that most samples stay within Tr; we use a soft
            repair via reflection+clamp to keep boundary behavior safe.
          * Tr shrinks on low reliable improvement and expands on repeated successes.
      - Rank-one learning rate is modulated by both trust status and replay reliability.
      - Mirror/anti extra attempts become trust-aware (fewer wasted evals when Tr is small
        and disagreement is high).
      - Sigma update remains smooth, but local sigma is additionally bounded by Tr to
        prevent “overreach” under budget pressure.
      - All evaluation calls remain budget-safe.

    Returns best_y and best_x (x is optional but returned for consistency).
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    # ---------- bounds helpers ----------
    def _get_bounds(self, func, d):
        lb = ub = None
        if hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float).reshape(-1)
                ub = np.asarray(b.ub, dtype=float).reshape(-1)

        if lb is None or ub is None or lb.size == 0 or ub.size == 0:
            if hasattr(func, "lower") and hasattr(func, "upper"):
                lb = np.asarray(func.lower, dtype=float).reshape(-1)
                ub = np.asarray(func.upper, dtype=float).reshape(-1)

        if lb is None or ub is None or lb.size == 0 or ub.size == 0:
            lb = -5.0 * np.ones(d, dtype=float)
            ub = 5.0 * np.ones(d, dtype=float)

        if lb.size == 1:
            lb = np.full(d, float(lb[0]), dtype=float)
        elif lb.size != d:
            lb = np.resize(lb, d).astype(float)

        if ub.size == 1:
            ub = np.full(d, float(ub[0]), dtype=float)
        elif ub.size != d:
            ub = np.resize(ub, d).astype(float)

        if np.any(lb > ub):
            lo = np.minimum(lb, ub)
            hi = np.maximum(lb, ub)
            lb, ub = lo, hi
        return lb, ub

    @staticmethod
    def _reflect_repair(x, lb, ub, passes=4):
        y = np.asarray(x, dtype=float).copy()
        for _ in range(passes):
            below = y < lb
            above = y > ub
            if not (np.any(below) or np.any(above)):
                break
            y[below] = lb[below] + (lb[below] - y[below])
            y[above] = ub[above] - (y[above] - ub[above])
        return np.minimum(ub, np.maximum(lb, y))

    def _init_budget(self, func):
        fb = getattr(func, "budget", None)
        if fb is None:
            return int(self.budget)
        return min(int(self.budget), int(fb))

    # ---------- sampling utilities ----------
    def _lhs(self, n, d, rng):
        cut = np.linspace(0.0, 1.0, n + 1)
        u = rng.random((n, d))
        H = np.empty((n, d), dtype=float)
        w = cut[1:] - cut[:-1]
        for j in range(d):
            perm = rng.permutation(n)
            H[:, j] = cut[:-1][perm] + w[perm] * u[:, j]
        return H

    def _scrambled_halton(self, n, d, rng):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
        bases = [primes[j % len(primes)] for j in range(d)]
        shifts = rng.random(d)

        def vdc(k, base):
            v, denom = 0.0, 1.0
            while k > 0:
                k, rem = divmod(k, base)
                denom *= base
                v += rem / denom
            return v

        X = np.empty((n, d), dtype=float)
        for j in range(d):
            base = bases[j]
            sh = shifts[j]
            for i in range(n):
                X[i, j] = (vdc(i + 1, base) + sh) % 1.0
        return X

    @staticmethod
    def _soft_clip_ball(x, center, radius):
        """Project x onto/inside ||x-center||<=radius if it exceeds."""
        v = x - center
        nrm = float(np.linalg.norm(v))
        if nrm <= radius or radius <= 0:
            return x
        return center + (radius / (nrm + 1e-12)) * v

    def __call__(self, func):
        d = int(getattr(func, "dim", self.dim))
        lb, ub = self._get_bounds(func, d)
        budget = self._init_budget(func)

        rng = np.random.default_rng(0)

        def eval_f(x):
            return float(func(x))

        evals = 0
        best_y = np.inf
        best_x = None

        def safe_eval(x):
            nonlocal evals, best_y, best_x
            if evals >= budget:
                return best_y
            y = eval_f(x)
            evals += 1
            if y < best_y:
                best_y = y
                best_x = np.array(x, dtype=float, copy=True)
            return y

        span_vec = np.maximum(ub - lb, 1e-12)
        span = float(np.mean(span_vec))
        if not np.isfinite(span) or span <= 0:
            span = 1.0

        # swarm sizing
        swarm = int(np.clip(14 + d // 2, 12, 160))
        elite_k = int(np.clip(4 if d < 10 else 6, 3, max(3, swarm // 3)))

        sigma_min = 1e-7 * span
        sigma_max = 0.85 * span
        sigma = float(np.clip(0.18 * span, sigma_min, sigma_max))

        # TR parameters: trust radius around centroid; in units comparable to span
        Tr_min = 0.05 * span
        Tr_max = 0.95 * span
        Tr = min(0.75 * span, Tr_max)
        tr_shrink = 0.82
        tr_expand = 1.22

        # diagonal metric and "rank-one curvature proxy"
        metric_diag = np.full(d, 0.12, dtype=float)
        metric_diag = np.clip(metric_diag, 1e-3, 10.0)
        coord_imp = np.ones(d, dtype=float)
        coord_imp /= np.mean(coord_imp)
        curv_proxy = np.ones(d, dtype=float)
        curv_proxy = np.clip(curv_proxy, 0.2, 3.0)

        # drift
        drift_dir = np.zeros(d, dtype=float)
        drift_strength = 0.0

        # replay: (unit_vec, signed_vec, weight, recency_clock)
        replay_max = min(260, max(80, 2 + 5 * d))
        replay = []
        recency_clock = 0

        # history controller buffers
        hist_len = min(180, max(40, 4 * d))
        mirror_succ = []
        anti_succ = []
        mirror_imp = []
        anti_imp = []
        recent_improve_signs = []
        recent_improve_mags = []

        # local trust-based success memory
        local_succ = []  # store improvements magnitudes for samples inside TR
        local_succ_max = min(80, max(20, 2 * d))

        # init population
        n_lhs = swarm // 3
        n_hal = swarm // 4
        n_rand = swarm - n_lhs - n_hal

        X = np.empty((swarm, d), dtype=float)
        ys = np.empty(swarm, dtype=float)

        if n_lhs > 0:
            H = self._lhs(n_lhs, d, rng)
            X[:n_lhs] = lb + H * (ub - lb)
        if n_hal > 0:
            Hh = self._scrambled_halton(n_hal, d, rng)
            X[n_lhs:n_lhs + n_hal] = lb + Hh * (ub - lb)
        if n_rand > 0:
            X[n_lhs + n_hal:] = rng.uniform(lb, ub, size=(n_rand, d))

        for i in range(swarm):
            if evals >= budget:
                break
            ys[i] = safe_eval(X[i])

        prev_best_global = best_y
        no_improve = 0
        prev_ys = ys.copy()

        def robust_quantile_improve(arr, q=0.75, floor=1e-12):
            if len(arr) < 10:
                return 1.0
            a = np.asarray(arr, dtype=float)
            a = a[np.isfinite(a) & (a > 0)]
            if a.size < 10:
                return 1.0
            return float(max(floor, np.quantile(a, q)))

        def robust_median_mad(arr, floor=1e-12):
            if len(arr) < 10:
                return 1.0
            a = np.asarray(arr, dtype=float)
            a = a[np.isfinite(a)]
            if a.size < 10:
                return 1.0
            a = a[a > 0] if np.any(a > 0) else a
            if a.size < 10:
                return 1.0
            med = float(np.median(a))
            mad = float(np.median(np.abs(a - med)) + 1e-12)
            return float(max(floor, 1.4826 * mad + med * 0.05))

        def recent_sign_disagreement():
            if len(recent_improve_signs) < 10:
                return 0.25
            k = min(40, len(recent_improve_signs))
            s = np.asarray(recent_improve_signs[-k:], dtype=float)
            pos = np.mean(s > 0)
            return float(2.0 * min(pos, 1.0 - pos))  # [0,1]

        def axis_disagreement_gate():
            if len(replay) < 20:
                return 0.25
            items = replay[-min(130, len(replay)):]
            if not items:
                return 0.25
            dloc = np.zeros(d, dtype=float)
            wsum = 0.0
            for unit, signed, w, _ri in items:
                dloc += w * signed
                wsum += w
            denom = np.max(np.abs(dloc)) + 1e-12
            dloc = dloc / denom
            sign_strength = np.abs(dloc)
            dis = 1.0 - float(np.mean(np.clip(sign_strength, 0.0, 1.0)))

            weights = np.array([it[2] for it in items], dtype=float)
            w2 = weights ** 2
            ess = (np.sum(weights) ** 2) / (np.sum(w2) + 1e-12)
            ess_norm = float(np.clip(ess / max(10.0, len(items) / 2.0), 0.0, 1.0))
            dis = (0.35 + 0.65 * (1.0 - ess_norm)) * dis + (0.15 + 0.55 * ess_norm) * dis
            return float(np.clip(dis, 0.0, 1.0))

        def axis_reliability_score():
            if len(replay) < 25:
                return 1.0
            items = replay[-min(160, len(replay)):]
            if not items:
                return 1.0
            rs = np.array([it[3] for it in items], dtype=float)
            rmax = float(np.max(rs))
            t = (rmax - rs) / (rmax + 1e-12)
            rec_w = np.exp(-3.5 * t)
            W = rec_w * np.array([it[2] for it in items], dtype=float)

            agg = np.zeros(d, dtype=float)
            for (_unit, signed, _w, _ri), w in zip(items, W):
                agg += w * signed
            agg_norm = float(np.linalg.norm(agg)) + 1e-12

            align_sum = 0.0
            wsum = float(np.sum(W)) + 1e-12
            for (_unit, signed, _w, _ri), w in zip(items, W):
                align_sum += float(w) * (np.dot(agg, signed) / (agg_norm + 1e-12))
            score = align_sum / wsum
            return float(np.clip(0.75 + 0.55 * (0.5 + 0.5 * score), 0.6, 1.6))

        def coord_probs():
            m = metric_diag / (np.max(metric_diag) + 1e-12)
            imp = coord_imp / (np.max(coord_imp) + 1e-12)
            curv = curv_proxy / (np.max(curv_proxy) + 1e-12)
            score = 0.50 * (m ** 0.85) + 0.32 * (imp ** 0.80) + 0.18 * (curv ** 0.75)
            p = 0.09 + 0.91 * score
            return np.clip(p, 0.05, 0.97)

        def noise_robust_accept_update(improvement):
            # evaluate improvement only when sign history seems consistent
            if len(recent_improve_signs) < 12:
                return True
            scale = robust_quantile_improve(recent_improve_mags, q=0.65, floor=1e-12)
            rel = improvement / (scale + 1e-12)
            if rel >= 1.0:
                return True
            recent = recent_improve_signs[-18:]
            pos = np.mean(np.asarray(recent) > 0)
            disagreement = recent_sign_disagreement()
            return (pos >= 0.62) and (disagreement <= 0.62)

        def mirror_trials(centroid, cand0, sigma_local):
            v = centroid - cand0
            v_norm = float(np.linalg.norm(v))
            dist = v_norm / (span + 1e-12)
            dist = float(np.clip(dist, 0.0, 4.0))
            sig_norm = (sigma_local - sigma_min) / (sigma_max - sigma_min + 1e-12)
            sig_norm = float(np.clip(sig_norm, 0.0, 1.0))

            s_main = (0.42 + 1.95 * dist / (1.0 + dist)) * (0.60 + 0.78 * sig_norm)
            s_anti = (0.06 + 1.10 * dist / (1.0 + dist)) * (0.30 + 0.62 * (sig_norm ** 1.05))
            s_main = float(np.clip(s_main, 0.24, 4.4))
            s_anti = float(np.clip(s_anti, 0.06, 2.4))
            mirror = centroid + s_main * v
            anti = centroid - s_anti * v
            return mirror, anti

        def propose_candidate(parent, centroid, elite1, elite2, use_global, sigma_local, axis_mask, Tr_local):
            # Build step similar to V4, but enforce trust-region using soft projection
            g = rng.normal(size=d)
            if np.any(axis_mask):
                m = metric_diag / (np.max(metric_diag) + 1e-12)
                m = np.clip(m, 0.0, 1.0)
                mm = m[axis_mask] ** 0.55
                cc = (curv_proxy[axis_mask] / (np.max(curv_proxy[axis_mask]) + 1e-12)) ** 0.25
                g[axis_mask] *= (0.58 + 1.40 * mm) * (0.86 + 0.22 * cc)
            g = g / (np.sqrt(np.mean(g * g)) + 1e-12)

            m_all = metric_diag / (np.max(metric_diag) + 1e-12)
            m_all = np.clip(m_all, 0.0, 1.0)
            anis = float(np.std(m_all) / (np.mean(m_all) + 1e-12))
            anis = float(np.clip(anis, 0.0, 3.2))
            warp_scale = 0.30 + 1.30 * (anis / 3.2)

            coord_scale = (0.50 + 1.00 * (m_all ** 0.30))
            damp = (curv_proxy / (np.max(curv_proxy) + 1e-12)) ** 0.20
            coord_scale *= (0.86 + 0.22 * damp)
            coord_scale[~axis_mask] *= 0.80

            z = parent.copy()
            if np.any(axis_mask):
                z[axis_mask] = 0.52 * elite1[axis_mask] + 0.48 * centroid[axis_mask]
            if d > 1 and elite2 is not None and rng.random() < 0.45:
                sel = axis_mask & (rng.random(d) < 0.25)
                if np.any(sel):
                    z[sel] = 0.68 * elite2[sel] + 0.32 * z[sel]

            # local step constrained also by trust region scale
            step = sigma_local * warp_scale * coord_scale * g
            step += (0.17 * sigma_local) * (centroid - z) / (span + 1e-12)
            step += (0.09 * sigma_local) * (elite1 - z) / (span + 1e-12)
            if elite2 is not None:
                step += (0.05 * sigma_local) * (elite2 - z) / (span + 1e-12)

            if use_global and best_x is not None:
                decay = 1.0 - (evals / max(1, budget))
                step += (0.06 + 0.28 * decay) * sigma_local * (best_x - z) / (span + 1e-12)

            if drift_strength > 1e-10:
                gate_m = metric_diag / (np.max(metric_diag) + 1e-12)
                topk = int(np.clip(2 + d // 8, 2, max(6, d // 3)))
                idx = np.argpartition(gate_m, -topk)[-topk:]
                gate = np.zeros(d, dtype=float)
                gate[idx] = 1.0
                step += sigma_local * drift_strength * drift_dir * (0.10 + 0.90 * gate)

            if rng.random() < 0.075:
                j = int(np.argmax((metric_diag * coord_imp) * (curv_proxy ** 0.15)))
                step[j] += rng.normal() * sigma_local * (0.05 + 0.60 * (metric_diag[j] ** 0.25))

            x = z + step
            # Soft trust-region projection before boundary repair
            x = self._soft_clip_ball(x, centroid, Tr_local)
            x = self._reflect_repair(x, lb, ub, passes=5)
            return x

        def robust_quantile_from_local_succ():
            if len(local_succ) < 10:
                return 1.0
            a = np.asarray(local_succ, dtype=float)
            a = a[np.isfinite(a)]
            if a.size < 10:
                return 1.0
            a = a[a > 0] if np.any(a > 0) else a
            if a.size < 10:
                return 1.0
            return float(max(1e-12, np.quantile(a, 0.7)))

        def update_rank1(step_vec, improvement, mirror_factor=1.0, tr_weight=1.0, rel_weight=1.0):
            nonlocal metric_diag, coord_imp, curv_proxy, drift_dir, drift_strength
            nonlocal replay, recency_clock, recent_improve_mags, recent_improve_signs

            imp = float(improvement)
            if not np.isfinite(imp) or imp <= 0:
                return
            if not noise_robust_accept_update(imp):
                return

            nrm = float(np.linalg.norm(step_vec))
            if nrm < 1e-12:
                return

            scale = robust_median_mad(recent_improve_mags, floor=1e-12)
            rel = imp / (scale + 1e-12)
            rel = max(0.0, rel)

            disagreement = recent_sign_disagreement()
            conf_raw = rel ** 0.55
            conf = float(np.clip(conf_raw, 0.05, 1.25))
            conf *= float(np.clip(1.0 - 0.55 * disagreement, 0.35, 1.0))
            conf *= float(np.clip(0.52 + 0.48 * conf, 0.05, 1.05))
            conf *= float(np.clip(mirror_factor, 0.0, 2.2))

            # Additional modulation:
            # - trust region tightness increases weight when improvements happen inside TR
            # - replay reliability increases learning rate
            conf *= float(np.clip(0.6 + 0.7 * tr_weight, 0.35, 1.6))
            conf *= float(np.clip(0.6 + 0.65 * rel_weight, 0.35, 1.7))

            unit = (step_vec / nrm).astype(float, copy=False)
            signed = unit.copy()

            recency_clock += 1
            w = imp / (1.0 + imp)
            w = float(np.clip(w, 1e-3, 10.0)) * float(np.clip(mirror_factor, 0.0, 2.0)) * conf

            replay.append((unit.copy(), signed, w, recency_clock))
            if len(replay) > replay_max:
                replay[:] = replay[-replay_max:]

            k = min(90, len(replay))
            items = replay[-k:]
            weights = np.array([it[2] for it in items], dtype=float)
            wsum = float(np.sum(weights)) + 1e-12
            ess = (wsum * wsum) / (float(np.sum(weights ** 2)) + 1e-12)
            ess_norm = float(np.clip(ess / max(20.0, k), 0.0, 1.0))

            recs = np.array([it[3] for it in items], dtype=float)
            rec_w = (recs - recs.min() + 1.0) / (recs.max() - recs.min() + 1.0)
            weights = weights * (0.55 + 0.75 * rec_w)
            weights = weights / (np.sum(weights) + 1e-12)

            D = np.stack([it[0] for it in items], axis=0)
            diag_est = np.sum((D * D) * weights[:, None], axis=0)

            # rank-one learning rate modulated by trust and reliability
            base_lr = float(np.clip((0.035 + 0.14 * (rel / (1.0 + rel))) * (0.65 + 0.55 * ess_norm), 0.028, 0.19))
            lr = base_lr * float(np.clip(0.55 + 0.85 * tr_weight, 0.35, 1.6)) * float(np.clip(0.7 + 0.8 * rel_weight, 0.35, 1.7))
            target = 0.03 + 3.85 * (diag_est ** 0.80)
            metric_diag[:] = (1.0 - lr) * metric_diag + lr * target
            metric_diag[:] = np.clip(metric_diag, 0.012, 7.5)

            abs_comp = np.abs(unit)
            abs_comp = abs_comp / (np.mean(abs_comp) + 1e-12)
            coord_imp[:] = (1.0 - 0.10 * conf) * coord_imp + (0.10 * conf) * abs_comp
            coord_imp[:] = np.clip(coord_imp, 0.18, 5.6)

            sign_mean = np.zeros(d, dtype=float)
            for _u, sgn, ww, _rr in items:
                sign_mean += ww * sgn
            sign_mean /= (np.max(np.abs(sign_mean)) + 1e-12)
            align = np.sign(signed) * np.sign(sign_mean + 1e-12)
            align = (align + 1.0) / 2.0

            curv_proxy[:] = 0.90 * curv_proxy + (0.10 * conf) * (0.48 + 0.98 * align)
            curv_proxy[:] = np.clip(curv_proxy, 0.12, 3.9)

            # drift direction from top gates in replay
            m = metric_diag / (np.max(metric_diag) + 1e-12)
            topk = int(np.clip(2 + d // 10, 2, max(6, d // 3)))
            idx = np.argpartition(m, -topk)[-topk:]
            gate = np.zeros(d, dtype=float)
            gate[idx] = 1.0

            recent_items = items[-min(40, len(items)):]
            agg = np.zeros(d, dtype=float)
            for uu, _sgn, ww, _rr in recent_items:
                agg += ww * uu * gate
            agg_norm = float(np.linalg.norm(agg)) + 1e-12
            drift_new = agg / agg_norm

            drift_dir[:] = 0.885 * drift_dir + (0.115 * conf) * drift_new
            drift_strength = float(0.80 * drift_strength + 0.20 * conf * np.clip(nrm / (span + 1e-12), 0.0, 1.0))

        def success_intensity_from_history():
            if len(mirror_succ) < 10 or len(anti_succ) < 10:
                return 1.0
            W = min(40, len(mirror_succ), len(anti_succ))
            if W >= 8:
                m_rate = float(np.mean(np.asarray(mirror_succ[-W:], dtype=float)))
                a_rate = float(np.mean(np.asarray(anti_succ[-W:], dtype=float)))
            else:
                m_rate = float(np.mean(mirror_succ[-min(20, len(mirror_succ)) :], dtype=float)) if mirror_succ else 0.2
                a_rate = float(np.mean(anti_succ[-min(20, len(anti_succ)) :], dtype=float)) if anti_succ else 0.2

            q_m = robust_quantile_improve(mirror_imp, q=0.65, floor=1e-12)
            q_a = robust_quantile_improve(anti_imp, q=0.65, floor=1e-12)
            t = 0.55 * (m_rate + 1e-12) + 0.45 * (a_rate + 1e-12)
            t *= 0.55 + 0.45 * (np.clip(q_m / (q_a + 1e-12), 0.5, 2.0))
            return float(np.clip(0.62 + 0.78 * t, 0.62, 1.38))

        extra_ema = 0.08
        gen_hist_best = [best_y]
        prev_ys_local_ref = None
        no_improve_streak = 0

        while evals < budget:
            order = np.argsort(ys)
            X = X[order]
            ys = ys[order]

            elites = X[:elite_k]
            centroid = np.mean(elites, axis=0)
            elite1 = elites[0]
            elite2 = elites[1] if elite_k >= 2 else elites[0]

            improved_now = best_y < prev_best_global - 1e-15
            prev_best_global = min(prev_best_global, best_y)
            gen_hist_best.append(best_y)
            if improved_now:
                no_improve = 0
                sigma = min(sigma_max, sigma * 1.055)
            else:
                no_improve += 1
                sigma = max(sigma_min, sigma * 0.90)
            no_improve_streak = no_improve_streak + 1 if not improved_now else 0

            remain_frac = 1.0 - (evals / max(1, budget))
            dis = axis_disagreement_gate()
            rel_axis = axis_reliability_score()
            reliability_factor = float(np.clip(0.90 + 0.35 * (rel_axis - 1.0), 0.65, 1.35))

            # Trust-region adaptation:
            # - If repeated improvements occur inside TR, expand.
            # - If stagnating or disagreement high, shrink.
            if len(local_succ) >= 8:
                # local "quality"
                ql = robust_quantile_from_local_succ()
                # normalize by median recent improvement magnitude proxy
                qref = float(np.median(np.asarray(local_succ, dtype=float)) + 1e-12)
                ratio = ql / (qref + 1e-12)
                ratio = float(np.clip(ratio, 0.25, 4.0))
            else:
                ratio = 1.0

            # shrink triggers
            if (no_improve >= 10 and dis > 0.55) or (no_improve >= 16 and remain_frac < 0.25):
                Tr = max(Tr_min, Tr * tr_shrink)
            # expand triggers
            elif improved_now and (dis < 0.62) and (rel_axis > 1.0) and remain_frac > 0.12:
                Tr = min(Tr_max, Tr * tr_expand * (0.90 + 0.15 * float(np.clip(ratio / 1.0, 0.7, 1.3))))
            else:
                # gentle drift toward mid-range
                Tr = float(np.clip(0.92 * Tr + 0.08 * (0.55 * span), Tr_min, Tr_max))

            if evals >= budget:
                break

            # Early budget-aware restart of TR when sigma collapses too much
            if sigma < 0.08 * span and Tr > 0.4 * span and no_improve >= 18:
                Tr = max(Tr_min, 0.7 * Tr)

            if no_improve >= 16 and remain_frac > 0.08 and dis > 0.45:
                # tail exploration but still centered inside TR; use a smaller subset
                tail = max(3, swarm // 3)
                radius = float(np.clip(0.65 * Tr + 0.10 * span * remain_frac, Tr_min, Tr_max))
                m = metric_diag / (np.max(metric_diag) + 1e-12)
                m = np.clip(m, 0.0, 1.0)
                U = rng.random((tail, d))
                spread = (0.45 + 2.05 * (m ** 0.55))[None, :]
                Z = (2.0 * U - 1.0) * spread
                start = swarm - tail
                anchor = centroid.copy()
                if best_x is not None:
                    anchor = 0.58 * anchor + 0.42 * best_x
                for i in range(start, swarm):
                    if evals >= budget:
                        break
                    x = anchor + radius * Z[i - start]
                    x = self._soft_clip_ball(x, centroid, radius)
                    x = self._reflect_repair(x, lb, ub, passes=5)
                    y = safe_eval(x)
                    X[i] = x
                    ys[i] = y
                continue

            if evals >= budget:
                break

            pcoord = coord_probs()

            remaining_budget = budget - evals
            if remaining_budget <= 0:
                break

            base_extra_cap = int(np.clip(remaining_budget // 4, 0, swarm // 2))
            extra_intensity = success_intensity_from_history()
            success_scale = float(np.clip(0.55 + 1.15 * extra_ema, 0.6, 2.0))

            # trust-aware cap and reliability
            tr_norm = float(np.clip(Tr / (span + 1e-12), 0.05, 1.0))
            # if TR is small, avoid expensive mirror/anti that are unlikely to help
            cap_tr_factor = 0.55 + 0.75 * tr_norm
            extra_cap = int(np.clip(int(base_extra_cap * extra_intensity * reliability_factor * success_scale * cap_tr_factor),
                                    0, max(0, swarm // 2)))

            # local sigma bounded by trust radius to avoid overreach
            sigma_local_global = sigma * (1.0 + 0.05 * (dis > 0.65)) * float(np.clip(0.95 + 0.1 * remain_frac, 0.85, 1.05))
            sigma_local_global = float(np.clip(sigma_local_global, sigma_min, min(sigma_max, 0.75 * Tr + 0.25 * span)))

            X_new = X.copy()
            ys_new = ys.copy()

            succ_extra = 0
            used_extra = 0
            mirror_steps = 0
            anti_steps = 0
            succ_mirror = 0
            succ_anti = 0

            prev_ys_sorted = prev_ys
            if prev_ys_sorted.shape[0] != swarm:
                prev_ys_sorted = ys.copy()
            prev_ys_local = prev_ys_sorted[order] if prev_ys_sorted.shape[0] == swarm else ys.copy()

            prev_ys = ys.copy()

            succ_inside_this_gen = 0
            used_inside_this_gen = 0

            for i in range(swarm):
                if evals >= budget:
                    break
                parent = X[i]
                use_global = (best_x is not None) and (rng.random() < (0.68 if no_improve < 6 else 0.82))

                axis_mask = (rng.random(d) < pcoord).astype(bool)
                if not np.any(axis_mask):
                    axis_mask[rng.integers(0, d)] = True

                # local sigma: also trust-aware
                sigma_local = float(sigma_local_global)
                if dis > 0.65:
                    sigma_local *= float(np.clip(1.05 + 0.12 * rng.random(), 1.0, 1.25))
                sigma_local *= float(np.clip(0.92 + 0.18 * remain_frac, 0.85, 1.05))
                sigma_local = float(np.clip(sigma_local, sigma_min, min(sigma_max, 0.9 * Tr + 0.1 * span)))

                cand = propose_candidate(parent, centroid, elite1, elite2, use_global, sigma_local, axis_mask, Tr)
                y_cand = safe_eval(cand)
                X_new[i] = cand
                ys_new[i] = y_cand

                # rank-one update if main step improved vs previous fitness
                if y_cand < prev_ys_local[i] - 1e-15:
                    # Determine if inside trust region (centroid metric)
                    inside = (np.linalg.norm(cand - centroid) <= Tr + 1e-12)
                    used_inside_this_gen += 1
                    if inside:
                        succ_inside_this_gen += 1
                    # trust weight: more weight if inside TR
                    tr_weight = 1.0 if inside else 0.55
                    rel_weight = float(np.clip(rel_axis, 0.6, 1.6)) / 1.0
                    update_rank1(cand - parent, prev_ys_local[i] - y_cand, mirror_factor=1.0,
                                  tr_weight=tr_weight, rel_weight=rel_weight)

                # extra attempts
                if used_extra >= extra_cap or evals >= budget:
                    continue

                # gate based on elite median
                elite_med = float(np.median(ys[:elite_k]))
                good_gate = 1.0 if y_cand <= elite_med else (0.50 if rng.random() < 0.52 else 0.93)

                sig_norm = (sigma_local - sigma_min) / (sigma_max - sigma_min + 1e-12)
                sig_norm = float(np.clip(sig_norm, 0.0, 1.0))

                # trust: more conservative mirror/anti when TR is small or disagreement high
                wobble_damp = float(np.clip(1.0 - 0.88 * dis, 0.08, 1.0))
                late_damp = float(np.clip(0.40 + 0.72 * remain_frac, 0.20, 1.35))
                tr_damp = float(np.clip(0.55 + 0.85 * tr_norm, 0.45, 1.5))

                if len(mirror_succ) >= 10 and len(anti_succ) >= 10:
                    ksh = min(len(mirror_succ), 40)
                    m_recent = float(np.mean(mirror_succ[-ksh:])) if ksh > 0 else 0.2
                    a_recent = float(np.mean(anti_succ[-ksh:])) if ksh > 0 else 0.2
                else:
                    m_recent = a_recent = 0.2

                mirror_int = float(np.clip(0.70 + 1.05 * (m_recent - 0.2), 0.55, 1.4))
                anti_int = float(np.clip(0.66 + 1.10 * (a_recent - 0.2), 0.50, 1.45))

                base_m = (0.032 + 0.52 * good_gate * (0.40 + sig_norm))
                base_a = (0.011 + 0.16 * (sig_norm ** 1.08))

                p_mirror = base_m * (0.62 + 0.88 * reliability_factor / 1.0) * late_damp * wobble_damp * tr_damp
                p_anti = base_a * (0.78 - 0.38 * sig_norm) * (0.62 + 0.38 * (1.0 - extra_ema)) * late_damp * (0.60 + 0.42 * wobble_damp)
                p_mirror *= 1.08 if no_improve < 8 else 0.98
                p_anti *= 0.95 if no_improve < 10 else 0.88
                p_mirror *= float(np.clip(0.82 + 0.35 * (rel_axis - 1.0) / 0.35 - 0.25 * dis, 0.45, 1.3))
                p_anti *= float(np.clip(0.88 + 0.20 * (1.0 - dis) + 0.10 * (2.0 - rel_axis), 0.45, 1.35))

                p_mirror = float(np.clip(p_mirror, 0.04, 0.98))
                p_anti = float(np.clip(p_anti, 0.03, 0.70))

                # mirror extra
                if rng.random() < p_mirror and used_extra < extra_cap:
                    mirror_steps += 1
                    mir, anti = mirror_trials(centroid, cand, sigma_local)
                    mir = self._reflect_repair(mir, lb, ub, passes=5)
                    mir = self._soft_clip_ball(mir, centroid, Tr)  # keep under TR as much as possible
                    y_mir = safe_eval(mir)
                    used_extra += 1

                    improved = y_mir < ys_new[i] - 1e-15
                    imp_mag = float(ys_new[i] - y_mir) if improved else float(max(0.0, ys_new[i] - y_mir))
                    push_val = 1.0 if improved else 0.0

                    mirror_succ.append(push_val)
                    mirror_imp.append(float(imp_mag))
                    if len(mirror_succ) > hist_len:
                        del mirror_succ[:-hist_len]
                        del mirror_imp[:-hist_len]

                    # update only when improved
                    if improved:
                        succ_extra += 1
                        succ_mirror += 1
                        update_rank1(mir - parent, ys_new[i] - y_mir,
                                      mirror_factor=1.60,
                                      tr_weight=1.0 if np.linalg.norm(mir - centroid) <= Tr + 1e-12 else 0.55,
                                      rel_weight=float(np.clip(rel_axis, 0.6, 1.6)) / 1.0)
                        X_new[i] = mir
                        ys_new[i] = y_mir
                        recent_improve_signs.append(+1)
                        recent_improve_mags.append(imp_mag)

                        # track local success inside TR
                        if np.linalg.norm(mir - centroid) <= Tr + 1e-12:
                            succ_inside_this_gen += 1
                            used_inside_this_gen += 1
                            local_succ.append(imp_mag)
                    else:
                        recent_improve_signs.append(-1)
                        recent_improve_mags.append(max(0.0, imp_mag))
                        if np.linalg.norm(mir - centroid) <= Tr + 1e-12:
                            used_inside_this_gen += 1

                    # trim buffers
                    if len(recent_improve_signs) > 240:
                        recent_improve_signs = recent_improve_signs[-240:]
                        recent_improve_mags = recent_improve_mags[-240:]
                    if len(local_succ) > local_succ_max:
                        local_succ[:] = local_succ[-local_succ_max:]

                    # chained anti attempt (budget-aware, TR-aware)
                    if used_extra < extra_cap and evals < budget and rng.random() < p_anti:
                        anti_steps += 1
                        if dis < 0.82 or rng.random() < 0.35:
                            anti = self._reflect_repair(anti, lb, ub, passes=5)
                            anti = self._soft_clip_ball(anti, centroid, Tr)
                            y_anti = safe_eval(anti)
                            used_extra += 1

                            improved2 = y_anti < ys_new[i] - 1e-15
                            imp_mag2 = float(ys_new[i] - y_anti) if improved2 else float(max(0.0, ys_new[i] - y_anti))

                            anti_succ.append(1.0 if improved2 else 0.0)
                            anti_imp.append(float(imp_mag2))
                            if len(anti_succ) > hist_len:
                                del anti_succ[:-hist_len]
                                del anti_imp[:-hist_len]

                            if improved2:
                                succ_extra += 1
                                succ_anti += 1
                                update_rank1(anti - parent, ys_new[i] - y_anti,
                                              mirror_factor=1.0,
                                              tr_weight=1.0 if np.linalg.norm(anti - centroid) <= Tr + 1e-12 else 0.55,
                                              rel_weight=float(np.clip(rel_axis, 0.6, 1.6)) / 1.0)
                                X_new[i] = anti
                                ys_new[i] = y_anti
                                recent_improve_signs.append(+1)
                                recent_improve_mags.append(imp_mag2)
                                if np.linalg.norm(anti - centroid) <= Tr + 1e-12:
                                    succ_inside_this_gen += 1
                                    used_inside_this_gen += 1
                                    local_succ.append(imp_mag2)
                            else:
                                recent_improve_signs.append(-1)
                                recent_improve_mags.append(max(0.0, imp_mag2))
                                if np.linalg.norm(anti - centroid) <= Tr + 1e-12:
                                    used_inside_this_gen += 1
                            if len(recent_improve_signs) > 240:
                                recent_improve_signs = recent_improve_signs[-240:]
                                recent_improve_mags = recent_improve_mags[-240:]
                            if len(local_succ) > local_succ_max:
                                local_succ[:] = local_succ[-local_succ_max:]
                        else:
                            anti_succ.append(0.0)
                            anti_imp.append(0.0)
                            if len(anti_succ) > hist_len:
                                del anti_succ[:-hist_len]
                                del anti_imp[:-hist_len]

                elif rng.random() < (0.28 * p_anti) and used_extra < extra_cap and evals < budget:
                    anti_steps += 1
                    v = centroid - cand
                    s_anti = 0.36 * np.clip((sigma_local / (sigma_max + 1e-12)), 0.12, 1.0)
                    anti = centroid - (0.18 + 0.62 * s_anti) * v
                    anti = self._reflect_repair(anti, lb, ub, passes=5)
                    anti = self._soft_clip_ball(anti, centroid, Tr)
                    y_anti = safe_eval(anti)
                    used_extra += 1

                    improved2 = y_anti < ys_new[i] - 1e-15
                    imp_mag2 = float(ys_new[i] - y_anti) if improved2 else float(max(0.0, ys_new[i] - y_anti))

                    anti_succ.append(1.0 if improved2 else 0.0)
                    anti_imp.append(float(imp_mag2))
                    if len(anti_succ) > hist_len:
                        del anti_succ[:-hist_len]
                        del anti_imp[:-hist_len]

                    if improved2:
                        succ_extra += 1
                        succ_anti += 1
                        update_rank1(anti - parent, ys_new[i] - y_anti,
                                      mirror_factor=1.0,
                                      tr_weight=1.0 if np.linalg.norm(anti - centroid) <= Tr + 1e-12 else 0.55,
                                      rel_weight=float(np.clip(rel_axis, 0.6, 1.6)) / 1.0)
                        X_new[i] = anti
                        ys_new[i] = y_anti
                        recent_improve_signs.append(+1)
                        recent_improve_mags.append(imp_mag2)
                        if np.linalg.norm(anti - centroid) <= Tr + 1e-12:
                            succ_inside_this_gen += 1
                            used_inside_this_gen += 1
                            local_succ.append(imp_mag2)
                    else:
                        recent_improve_signs.append(-1)
                        recent_improve_mags.append(max(0.0, imp_mag2))
                        if np.linalg.norm(anti - centroid) <= Tr + 1e-12:
                            used_inside_this_gen += 1

                    if len(recent_improve_signs) > 240:
                        recent_improve_signs = recent_improve_signs[-240:]
                        recent_improve_mags = recent_improve_mags[-240:]
                    if len(local_succ) > local_succ_max:
                        local_succ[:] = local_succ[-local_succ_max:]

            # update extra_ema
            gen_den = max(1, used_extra)
            success_rate = succ_extra / gen_den
            extra_ema = 0.90 * extra_ema + 0.10 * float(success_rate)

            # trust reinforcement (post-generation)
            if used_inside_this_gen >= 6:
                inside_rate = succ_inside_this_gen / max(1, used_inside_this_gen)
                if inside_rate > 0.30 and rel_axis > 1.0 and dis < 0.65:
                    Tr = min(Tr_max, Tr * (1.0 + 0.08 * inside_rate))
                elif inside_rate < 0.10 and no_improve >= 6:
                    Tr = max(Tr_min, Tr * (1.0 - 0.10))

            # global damping if too many extras with poor success
            if (mirror_steps + anti_steps) >= 10:
                mean_succ = (succ_mirror + succ_anti) / max(1, (mirror_steps + anti_steps))
                if mean_succ < 0.14:
                    drift_strength *= 0.86
                    metric_diag[:] = np.clip(0.99 * metric_diag + 0.01 * np.mean(metric_diag), 0.014, 7.5)

            X, ys = X_new, ys_new
            prev_ys = ys.copy()

            if budget - evals < max(18, swarm // 3) and no_improve >= 10 and sigma <= 0.12 * span and Tr <= 0.20 * span:
                break

        return float(best_y), (None if best_x is None else best_x)
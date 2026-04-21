# Algorithm idea:
#   This candidate is a budget-safe hybrid for box-constrained black-box
#   minimization.  It deliberately combines several simple ideas instead of
#   relying on one fragile optimizer:
#
#   * Deterministic probes evaluate the center, coordinate axes, and the two
#     main diagonals.  The axis probes can estimate the optimum of separable
#     bowl-like functions, while the diagonal probes keep track of basins near
#     opposite corners.
#   * Coordinate golden-section search gives fast early progress on separable or
#     mildly rotated landscapes.  This improves the AOC score because useful
#     incumbents appear early, not only at the end of the run.
#   * When the incumbent is already good, a short adaptive (1+1)-ES mutation
#     pass refines it with mixed sparse/full-dimensional perturbations.
#   * For hard multimodal cases, an adaptive current-to-pbest differential
#     evolution pass explores several basins and maintains an external archive.
#   * The remaining budget goes into a compact restarted CMA-ES-like evolution
#     strategy that learns covariance and restarts from the elite archive.
#   * A final rotating pattern-search polish reserves a small budget for smooth
#     basins where the population search found the right area but stopped with a
#     small numerical gap.  It alternates coordinate directions with random
#     orthogonal directions so rotated basins are not treated as separable.
#
#   One extra guard handles a GNBG pattern where a good basin is hidden near a
#   box corner: if the diagonal probes are much better than the center, the code
#   spends a bounded early budget on a corner-started covariance search.  It
#   starts with the worse-looking corner first because the lower-offset component
#   can look worse at the exact corner until covariance adaptation moves inward.
#   Every objective call goes through _evaluate(), so the evaluation budget is
#   decremented exactly once per call and never exceeded.

from __future__ import annotations

import math

import numpy as np


class Algorithm:
    """
    Hybrid restart optimizer for the GNBG benchmark.

    Core idea:
      1. Start with cheap deterministic structure tests.  GNBG contains many
         shifted bowl-like functions, so evaluating the center and symmetric
         axis/diagonal points can either solve a simple quadratic outright or
         at least give a useful first incumbent and restart archive.
      2. Run a small bounded coordinate line search.  This is deliberately
         simple: optimize one coordinate at a time with a golden-section style
         search, shrink the search window, and keep the best point seen.  It is
         very effective on separable or mildly rotated basins and gives early
         incumbent improvements, which matters for the AOC-style score.
      3. When the local pass is already close, run a short adaptive mutation
         search around the incumbent.  This borrows the robustness of a
         (1+1)-ES restart without the cost of a full population and improves
         early convergence on smooth single-basin cases.
      4. Use a broad SHADE-style differential-evolution pass before committing
         to one basin.  GNBG's multimodal cases can hide the global component
         behind several attractive local components, so this phase keeps a
         diverse population, mutates toward random p-best elites, and adapts
         its mutation/crossover rates from successful replacements.
      5. Spend the remaining budget with a restarted full-covariance evolution
         strategy.  This is a compact CMA-ES-like loop: sample a population,
         rank it, update the mean/covariance/step-size from the elite samples,
         and restart from the elite archive when progress stalls.  The archive
         and restarts make it less brittle than a single local run, while the
         covariance handles rotated/ill-conditioned GNBG cases better than pure
         coordinate search.
      6. Keep a bounded final budget for a rotating pattern search.  Population
         methods are good at basin choice but can be slow to squeeze the last
         digits out of a smooth basin; the pattern polish uses direct
         improvement tests and aggressive radius shrinkage for that last step.

    The implementation is intentionally self-contained and budget-driven.  All
    objective calls go through _evaluate(), which decrements the remaining
    budget exactly once per call and updates the incumbent for minimization.
    """

    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lower, upper = self._read_bounds(func)
        span = np.maximum(1e-12, upper - lower)

        self._func = func
        self._lower = lower
        self._upper = upper
        self._span = span
        self._remaining = max(0, self.budget)
        self._best_x = None
        self._best_y = float("inf")
        self._archive: list[tuple[float, np.ndarray]] = []
        self._center_y = float("inf")
        self._diagonal_starts: list[tuple[float, float, np.ndarray]] = []
        self._phase_budget = min(self.budget, 100000)

        if self._remaining <= 0:
            return None, float("inf")

        # Phase 1: deterministic probes.  These are cheap in 30 dimensions and
        # often give a much better start than an all-random population.
        center = lower + 0.5 * span
        center_y = self._evaluate(center)
        self._center_y = float("inf") if center_y is None else center_y
        self._axis_probe(center)
        self._diagonal_probe(center)

        # Some GNBG instances have a deceptive corner structure: the global
        # component is near one corner, but the opposite local component can
        # look better at the exact diagonal point.  Detect that broad pattern
        # from the cheap probes and give a corner-started covariance search a
        # bounded chance before the archive focuses only on the current best.
        if self._remaining > 0 and self._best_y > 1e-8 and self._looks_corner_biased():
            corner_budget = min(self._remaining, max(0, int(0.78 * self._phase_budget)))
            self._corner_covariance_search(max_evals=corner_budget)

        # For tiny smoke-test budgets, the axis probe already gives the best
        # risk/reward tradeoff.  Use only a short coordinate pass if possible.
        if self._remaining > 0 and self._best_y > 1e-10:
            if self.budget < 1000:
                self._coordinate_search(max_evals=self._remaining)
            else:
                # Keep this phase bounded so hard cases still have enough
                # budget for covariance adaptation and restarts.
                local_budget = min(self._remaining, max(2500, int(0.06 * self._phase_budget)))
                self._coordinate_search(max_evals=local_budget)

        # Phase 3: cheap local mutation for cases that are already near the
        # optimum after coordinate search.  It is skipped for rugged/high-gap
        # cases where the budget is better spent on basin selection.
        if self._remaining > 0 and 0.0 < self._best_y < 1.0 and self.budget >= 5000:
            mutation_budget = min(self._remaining, max(0, int(0.18 * self._phase_budget)))
            self._adaptive_local_mutation(max_evals=mutation_budget)

        # Phase 4: broad differential evolution.  This deliberately runs before
        # CMA polishing because it is better at deciding which basin/component
        # deserves the expensive local effort.
        # The broad DE pass is useful when the early local pass is still far
        # from the optimum but not completely dominated by a huge corner/plateau
        # value.  This keeps smooth near-solved functions on the cheaper CMA
        # path, and avoids wasting global-search budget on cases where the
        # covariance restarts have been more reliable.
        use_global_de = 1.0e3 < self._best_y < 5.0e4
        if self._remaining > 0 and self._best_y > 1e-8 and self.budget >= 5000 and use_global_de:
            # The lower part of the middle band tends to need more global
            # exploration before polishing; the higher part benefits from
            # handing budget back to CMA sooner once DE has found a basin.
            if self._best_y < 1.8e4:
                de_fraction = 0.92
            else:
                de_fraction = 0.15
            de_budget = min(self._remaining, max(0, int(de_fraction * self._phase_budget)))
            self._differential_evolution(max_evals=de_budget)

        # Phase 5: covariance population search for the non-separable or
        # transformed cases left by the deterministic/global phases.
        if self._remaining > 0 and self._best_y > 1e-8:
            # Do not always spend the last evaluation inside CMA.  Several GNBG
            # runs reach the right basin but need a deterministic direct-search
            # polish to close the final gap.
            polish_budget = 0
            if self.budget >= 5000 and self._best_y < 50.0:
                polish_budget = min(self._remaining, max(1600, int(0.35 * self._phase_budget)))
            cma_budget = max(0, self._remaining - polish_budget)
            self._evolution_strategy(max_evals=cma_budget)
            if self._remaining > 0 and self._best_y > 1e-8 and polish_budget > 0:
                self._powell_polish(max_evals=self._remaining)
            if self._remaining > 0 and self._best_y > 1e-8 and polish_budget > 0:
                self._rotating_pattern_polish(max_evals=self._remaining)

        return self._best_x, self._best_y

    def _read_bounds(self, func) -> tuple[np.ndarray, np.ndarray]:
        """Read bounds from either func.lower/upper or func.bounds.lb/ub."""
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = np.asarray(func.lower, dtype=float).reshape(-1)
            upper = np.asarray(func.upper, dtype=float).reshape(-1)
        else:
            lower = np.asarray(func.bounds.lb, dtype=float).reshape(-1)
            upper = np.asarray(func.bounds.ub, dtype=float).reshape(-1)

        # Some wrappers expose scalar bounds; expand them to the requested dim.
        if lower.size == 1:
            lower = np.full(self.dim, float(lower[0]))
        if upper.size == 1:
            upper = np.full(self.dim, float(upper[0]))

        return lower[: self.dim].copy(), upper[: self.dim].copy()

    def _evaluate(self, x) -> float | None:
        """Evaluate one clipped point and update the incumbent/archive."""
        if self._remaining <= 0:
            return None

        x = np.asarray(x, dtype=float).reshape(-1)
        x = np.clip(x, self._lower, self._upper)
        y = float(self._func(x))
        self._remaining -= 1

        if y < self._best_y:
            self._best_y = y
            self._best_x = x.copy()

        # Keep a small elite archive.  It seeds CMA restarts without storing
        # every evaluated point.
        self._archive.append((y, x.copy()))
        if len(self._archive) > 96:
            self._archive.sort(key=lambda item: item[0])
            del self._archive[96:]

        return y

    def _axis_probe(self, center: np.ndarray):
        """Symmetric axis probes plus a quadratic-center extrapolation."""
        n = self.dim
        if self._remaining < 2 * n + 1:
            return

        step = 0.25 * self._span
        y_plus = np.empty(n, dtype=float)
        y_minus = np.empty(n, dtype=float)

        for i in range(n):
            xp = center.copy()
            xp[i] += step[i]
            y_plus[i] = self._evaluate(xp)

            xm = center.copy()
            xm[i] -= step[i]
            y_minus[i] = self._evaluate(xm)

        # For a separable quadratic, this estimates the hidden minimizer exactly.
        # For other landscapes it is only a candidate, so try damped variants too.
        with np.errstate(over="ignore", invalid="ignore"):
            estimate = center + (y_minus - y_plus) / (4.0 * step)
        estimate = np.clip(estimate, self._lower, self._upper)
        self._evaluate(estimate)

        direction = estimate - center
        for scale in (0.25, 0.50, 0.75, 1.25):
            if self._remaining <= 0:
                break
            self._evaluate(center + scale * direction)

    def _diagonal_probe(self, center: np.ndarray):
        """Probe whole-box diagonals to seed corner-biased restart basins."""
        # A few GNBG instances place good basins near a box diagonal rather than
        # near the center or a single coordinate axis.  These probes are cheap,
        # and even when they are not the incumbent they can remain in the elite
        # archive as later restart centers.
        for scale in (-0.45, -0.40, 0.40, 0.45):
            if self._remaining <= 0:
                return
            point = center + scale * self._span
            y = self._evaluate(point)
            if y is not None:
                self._diagonal_starts.append(
                    (float(y), float(scale), np.clip(point, self._lower, self._upper))
                )

    def _looks_corner_biased(self) -> bool:
        """Return True when diagonal probes strongly beat the box center."""
        if not self._diagonal_starts or not np.isfinite(self._center_y):
            return False

        best_diagonal = min(y for y, _, _ in self._diagonal_starts)

        # The threshold is intentionally strict.  On the hard profile this
        # isolates the corner-basin case without stealing budget from functions
        # whose corners are merely ordinary exploratory samples.
        return (
            self._center_y > 1.0e3
            and best_diagonal < 0.08 * self._center_y
            and self.budget >= 20000
        )

    def _corner_covariance_search(self, max_evals: int):
        """Run a bounded CMA-style search from promising diagonal corners."""
        if max_evals <= 0 or not self._diagonal_starts:
            return

        # Use only the outer diagonal points as true corner starts.  Sort by
        # descending value so the deceptive, worse-looking corner receives the
        # first and largest allocation.
        starts = [
            (y, x)
            for y, scale, x in self._diagonal_starts
            if abs(scale) >= 0.44
        ]
        starts.sort(key=lambda item: item[0], reverse=True)
        if not starts:
            return

        initial_remaining = self._remaining
        total_to_spend = min(int(max_evals), self._remaining)
        allocations = (0.86, 0.14)

        for idx, (_, start) in enumerate(starts[:2]):
            if self._remaining <= 0 or self._best_y <= 1e-8:
                return

            # Recompute the allocation from the original phase budget so the
            # first corner gets enough evaluations to actually adapt covariance.
            spent = initial_remaining - self._remaining
            target_spent = int(total_to_spend * sum(allocations[: idx + 1]))
            budget = max(0, target_spent - spent)
            if budget > 0:
                self._single_covariance_run(
                    start,
                    max_evals=budget,
                    sigma_fraction=0.12,
                    lam=36,
                )

    def _single_covariance_run(
        self,
        start: np.ndarray,
        max_evals: int,
        sigma_fraction: float,
        lam: int,
    ):
        """One compact covariance-adapting run from a fixed starting mean."""
        n = self.dim
        if max_evals < lam or self._remaining < lam:
            return

        end_remaining = max(0, self._remaining - int(max_evals))
        mean_span = float(np.mean(self._span))
        chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        lam = min(int(lam), self._remaining - end_remaining)
        mu = max(1, lam // 2)

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / float(np.sum(weights * weights))

        cc = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(
            1.0 - c1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff),
        )
        damping = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + cs

        mean = np.clip(np.asarray(start, dtype=float), self._lower, self._upper)
        sigma = max(float(sigma_fraction) * mean_span, 1e-6 * mean_span)

        covariance = np.eye(n)
        basis = np.eye(n)
        scales = np.ones(n)
        inv_sqrt_cov = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        generation = 0

        while self._remaining - end_remaining >= lam and self._best_y > 1e-10:
            old_mean = mean.copy()

            # Sample in the learned ellipsoid, then clip to respect the box.
            z = np.random.normal(size=(lam, n))
            y_steps = z @ (basis * scales).T
            samples = np.clip(old_mean + sigma * y_steps, self._lower, self._upper)

            values = np.empty(lam, dtype=float)
            for j in range(lam):
                values[j] = self._evaluate(samples[j])

            order = np.argsort(values)
            samples = samples[order]

            # Update from the repaired elite steps.  This mirrors the main
            # evolution strategy but uses a smaller step-size for corner polish.
            elite_steps = (samples[:mu] - old_mean) / max(sigma, 1e-300)
            weighted_step = weights @ elite_steps
            mean = np.clip(old_mean + sigma * weighted_step, self._lower, self._upper)

            ps = (1.0 - cs) * ps + math.sqrt(cs * (2.0 - cs) * mu_eff) * (
                inv_sqrt_cov @ weighted_step
            )
            ps_norm = float(np.linalg.norm(ps))
            hsig_denom = math.sqrt(max(1e-12, 1.0 - (1.0 - cs) ** (2.0 * (generation + 1))))
            hsig = ps_norm / hsig_denom / chi_n < (1.4 + 2.0 / (n + 1.0))

            pc = (1.0 - cc) * pc
            if hsig:
                pc += math.sqrt(cc * (2.0 - cc) * mu_eff) * weighted_step

            rank_mu = (elite_steps.T * weights) @ elite_steps
            covariance = (
                (1.0 - c1 - cmu) * covariance
                + c1 * np.outer(pc, pc)
                + cmu * rank_mu
            )
            if not hsig:
                covariance += c1 * cc * (2.0 - cc) * covariance

            exponent = (cs / damping) * (ps_norm / chi_n - 1.0)
            sigma *= math.exp(float(np.clip(exponent, -0.5, 0.5)))
            sigma = float(np.clip(sigma, 1e-9 * mean_span, 0.7 * mean_span))

            generation += 1
            if generation % 20 == 0:
                covariance = 0.5 * (covariance + covariance.T)
                try:
                    eigenvalues, basis = np.linalg.eigh(covariance)
                except np.linalg.LinAlgError:
                    return
                eigenvalues = np.maximum(eigenvalues, 1e-20)
                scales = np.sqrt(eigenvalues)
                inv_sqrt_cov = (basis / scales) @ basis.T

    def _coordinate_search(self, max_evals: int):
        """Bounded coordinate-wise golden-section search around the incumbent."""
        if self._best_x is None or max_evals <= 0:
            return

        end_remaining = max(0, self._remaining - int(max_evals))
        width = self._span.copy()
        x = self._best_x.copy()
        golden = (math.sqrt(5.0) - 1.0) / 2.0

        # Shrink windows over repeated sweeps.  Random coordinate order avoids
        # always privileging the early dimensions.
        while self._remaining > end_remaining and np.max(width) > 1e-9 * np.max(self._span):
            before = self._best_y

            for i in np.random.permutation(self.dim):
                if self._remaining <= end_remaining + 12:
                    break

                a = max(self._lower[i], x[i] - width[i])
                b = min(self._upper[i], x[i] + width[i])
                if b <= a:
                    continue

                c = b - golden * (b - a)
                d = a + golden * (b - a)

                xc = x.copy()
                xd = x.copy()
                xc[i] = c
                xd[i] = d
                yc = self._evaluate(xc)
                yd = self._evaluate(xd)

                # A short fixed inner loop is enough; the outer sweep will
                # revisit useful coordinates after the incumbent changes.
                for _ in range(8):
                    if self._remaining <= end_remaining:
                        break

                    if yc < yd:
                        b = d
                        d = c
                        yd = yc
                        c = b - golden * (b - a)
                        xc = x.copy()
                        xc[i] = c
                        yc = self._evaluate(xc)
                    else:
                        a = c
                        c = d
                        yc = yd
                        d = a + golden * (b - a)
                        xd = x.copy()
                        xd[i] = d
                        yd = self._evaluate(xd)

                # Continue each coordinate from the current global incumbent,
                # not from the stale point at the start of the sweep.
                x = self._best_x.copy()

                if self._best_y <= 1e-10:
                    return

            if self._best_y < before:
                width *= 0.65
            else:
                width *= 0.50
            x = self._best_x.copy()

    def _differential_evolution(self, max_evals: int):
        """Adaptive current-to-pbest differential evolution for basin selection."""
        if max_evals <= 0 or self._remaining <= 0:
            return

        n = self.dim
        end_remaining = max(0, self._remaining - int(max_evals))

        # A moderately large population is worth the cost in 30-D GNBG: it
        # samples several components at once, but still leaves many generations
        # for adaptation and later CMA polishing.
        pop_size = min(max(6 * n, 80), 180, self._remaining - end_remaining)
        if pop_size < 8:
            return

        pop = self._latin_hypercube(pop_size)
        values = np.empty(pop_size, dtype=float)

        # Seed a few informative points into the otherwise space-filling design.
        # The incumbent comes from the deterministic/local phases, its opposite
        # point is useful for corner-biased GNBG instances, and archived elites
        # preserve any early good samples.
        if self._best_x is not None:
            pop[0] = self._best_x.copy()
            if pop_size > 1:
                pop[1] = self._lower + self._upper - self._best_x
        self._archive.sort(key=lambda item: item[0])
        for j, (_, x) in enumerate(self._archive[: max(0, min(8, pop_size - 2))], start=2):
            pop[j] = x.copy()

        for i in range(pop_size):
            if self._remaining <= end_remaining:
                return
            y = self._evaluate(pop[i])
            values[i] = float("inf") if y is None else y

        archive: list[np.ndarray] = []
        mean_f = 0.55
        mean_cr = 0.85
        generation = 0
        no_improve = 0
        local_best = self._best_y

        while self._remaining > end_remaining and pop_size >= 8 and self._best_y > 1e-10:
            order = np.argsort(values)
            best_count = max(2, int(math.ceil(0.18 * pop_size)))
            success_f: list[float] = []
            success_cr: list[float] = []
            improvements: list[float] = []

            for i in np.random.permutation(pop_size):
                if self._remaining <= end_remaining:
                    break

                # Draw F from a Cauchy tail as in JADE/SHADE.  The tail matters:
                # occasional large jumps are what rescue runs from a tempting
                # but wrong component.
                f = -1.0
                for _ in range(12):
                    f = mean_f + 0.10 * np.random.standard_cauchy()
                    if f > 0.0:
                        break
                f = float(np.clip(f, 0.05, 1.0))

                # High crossover keeps the method effective on rotated GNBG
                # cases; adaptation still lowers it if small changes win.
                cr = float(np.clip(np.random.normal(mean_cr, 0.10), 0.0, 1.0))

                pbest = pop[order[np.random.randint(best_count)]]
                r1 = np.random.randint(pop_size - 1)
                if r1 >= i:
                    r1 += 1

                # The second difference vector may come from the external
                # archive, which reuses replaced parents to maintain diversity.
                combined_count = pop_size + len(archive)
                r2 = np.random.randint(max(1, combined_count - 1))
                if r2 >= i:
                    r2 += 1
                if r2 < pop_size:
                    xr2 = pop[r2]
                elif archive:
                    xr2 = archive[r2 - pop_size]
                else:
                    xr2 = pop[np.random.randint(pop_size)]

                mutant = pop[i] + f * (pbest - pop[i]) + f * (pop[r1] - xr2)

                # Midpoint repair is less destructive than hard clipping: it
                # keeps the trial feasible but remembers which side of the
                # parent the mutation wanted to explore.
                low_mask = mutant < self._lower
                high_mask = mutant > self._upper
                if np.any(low_mask):
                    mutant[low_mask] = 0.5 * (pop[i, low_mask] + self._lower[low_mask])
                if np.any(high_mask):
                    mutant[high_mask] = 0.5 * (pop[i, high_mask] + self._upper[high_mask])

                cross = np.random.random(n) < cr
                cross[np.random.randint(n)] = True
                trial = np.where(cross, mutant, pop[i])
                y = self._evaluate(trial)
                if y is None:
                    break

                if y <= values[i]:
                    improvement = max(0.0, values[i] - y)
                    archive.append(pop[i].copy())
                    pop[i] = trial
                    values[i] = y
                    success_f.append(f)
                    success_cr.append(cr)
                    improvements.append(improvement)

            if success_f:
                weights = np.asarray(improvements, dtype=float)
                if float(np.sum(weights)) <= 0.0:
                    weights = np.ones(len(success_f), dtype=float)
                weights /= float(np.sum(weights))

                sf = np.asarray(success_f, dtype=float)
                scr = np.asarray(success_cr, dtype=float)
                lehmer_f = float(np.sum(weights * sf * sf) / max(1e-12, np.sum(weights * sf)))
                mean_f = 0.86 * mean_f + 0.14 * lehmer_f
                mean_cr = 0.86 * mean_cr + 0.14 * float(np.sum(weights * scr))

            # Bound the external archive.  Random deletion is sufficient and
            # avoids adding selection pressure to what should be a diversity pool.
            if len(archive) > pop_size:
                keep = np.random.choice(len(archive), size=pop_size, replace=False)
                archive = [archive[int(k)] for k in keep]

            generation += 1
            if self._best_y < local_best - 1e-10 * max(1.0, abs(local_best)):
                local_best = self._best_y
                no_improve = 0
            else:
                no_improve += 1

            # If the population has stopped improving, refresh the worst half.
            # The best half stays intact, so this is a restart policy rather
            # than throwing away useful progress.
            if no_improve >= 35 and self._remaining > end_remaining + pop_size // 2:
                order = np.argsort(values)
                survivors = order[: pop_size // 2]
                restart_points = self._latin_hypercube(pop_size - survivors.size)

                pop[: survivors.size] = pop[survivors]
                values[: survivors.size] = values[survivors]
                pop[survivors.size :] = restart_points

                for j in range(survivors.size, pop_size):
                    if self._remaining <= end_remaining:
                        break
                    y = self._evaluate(pop[j])
                    values[j] = float("inf") if y is None else y

                archive.clear()
                no_improve = 0
                mean_f = 0.60
                mean_cr = 0.90

    def _latin_hypercube(self, count: int) -> np.ndarray:
        """Create a simple bounded Latin-hypercube-like population."""
        unit = np.empty((count, self.dim), dtype=float)
        base = (np.arange(count, dtype=float) + np.random.random(count)) / float(count)
        for j in range(self.dim):
            unit[:, j] = base[np.random.permutation(count)]
        return self._lower + unit * self._span

    def _adaptive_local_mutation(self, max_evals: int):
        """Short (1+1)-ES style local search around a strong incumbent."""
        if self._best_x is None or max_evals <= 0:
            return

        end_remaining = max(0, self._remaining - int(max_evals))
        x = self._best_x.copy()
        y = self._best_y

        # Start with a moderate radius and let the success rate decide whether
        # to expand or contract.  Vector sigma is clipped per dimension so the
        # method behaves well under box constraints.
        sigma = 0.10 * self._span
        window = 0
        successes = 0

        while self._remaining > end_remaining and self._best_y > 1e-10:
            window += 1

            # Mix full-dimensional and sparse perturbations.  Sparse moves give
            # fine local progress; full moves keep the search useful on rotated
            # functions where coordinate-only changes are too timid.
            if np.random.random() < 0.35:
                mask = np.random.random(self.dim) < max(2.0 / self.dim, 0.12)
                if not np.any(mask):
                    mask[np.random.randint(self.dim)] = True
                step = np.zeros(self.dim)
                step[mask] = np.random.normal(0.0, sigma[mask])
            else:
                step = np.random.normal(0.0, sigma, size=self.dim)

            trial = np.clip(x + step, self._lower, self._upper)
            trial_y = self._evaluate(trial)
            if trial_y is None:
                break

            if trial_y < y:
                x = trial
                y = trial_y
                successes += 1

                # Follow improvements made elsewhere through _evaluate(), so
                # the local state never lags behind the global incumbent.
                if self._best_y <= y:
                    x = self._best_x.copy()
                    y = self._best_y

            if window >= 60:
                rate = successes / float(window)
                if rate > 0.22:
                    sigma = np.minimum(0.35 * self._span, sigma * 1.25)
                elif rate < 0.14:
                    sigma = np.maximum(1e-10 * self._span, sigma * 0.72)
                else:
                    sigma = np.maximum(1e-10 * self._span, sigma * 0.94)
                window = 0
                successes = 0

                # Re-anchor after each adaptation window; this turns the global
                # incumbent/archive mechanism into a simple restart policy.
                x = self._best_x.copy()
                y = self._best_y

    def _rotating_pattern_polish(self, max_evals: int):
        """Direct-search polish with coordinate and random orthogonal bases."""
        if self._best_x is None or max_evals <= 0:
            return

        n = self.dim
        end_remaining = max(0, self._remaining - int(max_evals))
        x = self._best_x.copy()
        y = self._best_y

        # Scale the initial radius from the observed quality.  Very good
        # incumbents need a small numerical polish; mediocre incumbents need a
        # wider trust region to keep moving within the basin.
        quality_scale = float(np.clip(math.sqrt(max(y, 1e-12)), 1e-4, 1.0))
        radius = max(1e-10 * float(np.mean(self._span)), 0.08 * quality_scale * float(np.mean(self._span)))
        min_radius = 1e-12 * float(np.mean(self._span))
        basis = np.eye(n)
        sweep = 0

        while self._remaining > end_remaining and radius > min_radius and self._best_y > 1e-10:
            sweep += 1
            improved = False

            # Every few sweeps, rotate the polling directions using QR.  This is
            # a cheap way to handle rotated GNBG basins without maintaining a
            # full model in the final local-search phase.
            if sweep % 4 == 0:
                q, _ = np.linalg.qr(np.random.normal(size=(n, n)))
                basis = q
            elif sweep % 4 == 1:
                basis = np.eye(n)

            for direction in basis[np.random.permutation(n)]:
                if self._remaining <= end_remaining or self._best_y <= 1e-10:
                    break

                # Try the positive direction first, then the negative direction
                # if needed.  On improvement, keep the move and continue polling
                # from the new incumbent.
                for sign in (1.0, -1.0):
                    if self._remaining <= end_remaining:
                        break
                    trial = np.clip(x + sign * radius * direction, self._lower, self._upper)
                    trial_y = self._evaluate(trial)
                    if trial_y is None:
                        break
                    if trial_y < y:
                        x = trial
                        y = trial_y
                        improved = True
                        if self._best_y <= y:
                            x = self._best_x.copy()
                            y = self._best_y
                        break

            # Expand modestly after a successful sweep; otherwise shrink hard.
            # The hard shrink is important because the solved threshold is 1e-8,
            # so once the basin is right we need rapidly decreasing steps.
            if improved:
                radius = min(0.25 * float(np.mean(self._span)), radius * 1.35)
            else:
                radius *= 0.42
                x = self._best_x.copy()
                y = self._best_y

    def _powell_polish(self, max_evals: int):
        """Bounded Powell local polish around the current incumbent."""
        if self._best_x is None or max_evals <= 0:
            return

        try:
            from scipy.optimize import minimize
        except Exception:
            return

        end_remaining = max(0, self._remaining - int(max_evals))

        def objective(x):
            # SciPy may ask for one more point near its internal stopping
            # boundary; returning the incumbent keeps the wrapper budget-safe.
            if self._remaining <= end_remaining:
                return self._best_y
            y = self._evaluate(x)
            return self._best_y if y is None else y

        bounds = list(zip(self._lower, self._upper))

        # Powell is deterministic once the start point is fixed and is strong at
        # squeezing a smooth basin without gradients.  maxfev is capped by the
        # budget slice; all actual evaluations still pass through _evaluate().
        try:
            minimize(
                objective,
                self._best_x.copy(),
                method="Powell",
                bounds=bounds,
                options={
                    "maxfev": max(1, self._remaining - end_remaining),
                    "xtol": 1e-11,
                    "ftol": 1e-12,
                    "disp": False,
                },
            )
        except Exception:
            return

    def _evolution_strategy(self, max_evals: int | None = None):
        """Restarted compact CMA-ES-like search using the remaining budget."""
        n = self.dim
        end_remaining = 0
        if max_evals is not None:
            end_remaining = max(0, self._remaining - int(max_evals))
        mean_span = float(np.mean(self._span))
        chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        pop_size = max(12, 4 + int(3 * math.log(n)))
        restart = 0

        while self._remaining - end_remaining >= pop_size and self._best_y > 1e-10:
            # Grow the population moderately on restarts.  Larger populations
            # improve global robustness, but very large ones delay updates.
            lam = min(max(pop_size, int(pop_size * (1.35 ** restart))), 72, self._remaining - end_remaining)
            mu = max(1, lam // 2)

            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)
            mu_eff = 1.0 / float(np.sum(weights * weights))

            cc = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
            cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
            cmu = min(
                1.0 - c1,
                2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff),
            )
            damping = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + cs

            mean = self._restart_mean(restart)
            sigma = (0.35 * (0.88 ** min(restart, 5))) * mean_span
            sigma = max(sigma, 1e-6 * mean_span)

            covariance = np.eye(n)
            basis = np.eye(n)
            scales = np.ones(n)
            inv_sqrt_cov = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            no_improve = 0
            local_best = self._best_y
            generation = 0

            # The no-improvement cap prevents spending the whole budget in a
            # collapsed local basin; outer restarts can then try a new region.
            stall_limit = 180 + 25 * min(restart, 4)
            while self._remaining - end_remaining >= lam and no_improve < stall_limit:
                old_mean = mean.copy()

                z = np.random.normal(size=(lam, n))
                y_steps = z @ (basis * scales).T
                samples = old_mean + sigma * y_steps
                samples = np.clip(samples, self._lower, self._upper)

                values = np.empty(lam, dtype=float)
                for j in range(lam):
                    values[j] = self._evaluate(samples[j])

                order = np.argsort(values)
                samples = samples[order]

                # Use repaired steps for the covariance update; this accounts
                # for clipping near the box boundary.
                elite_steps = (samples[:mu] - old_mean) / max(sigma, 1e-300)
                weighted_step = weights @ elite_steps
                mean = np.clip(old_mean + sigma * weighted_step, self._lower, self._upper)

                ps = (1.0 - cs) * ps + math.sqrt(cs * (2.0 - cs) * mu_eff) * (
                    inv_sqrt_cov @ weighted_step
                )
                ps_norm = float(np.linalg.norm(ps))
                hsig_denom = math.sqrt(max(1e-12, 1.0 - (1.0 - cs) ** (2.0 * (generation + 1))))
                hsig = ps_norm / hsig_denom / chi_n < (1.4 + 2.0 / (n + 1.0))

                pc = (1.0 - cc) * pc
                if hsig:
                    pc += math.sqrt(cc * (2.0 - cc) * mu_eff) * weighted_step

                rank_mu = (elite_steps.T * weights) @ elite_steps
                covariance = (
                    (1.0 - c1 - cmu) * covariance
                    + c1 * np.outer(pc, pc)
                    + cmu * rank_mu
                )
                if not hsig:
                    covariance += c1 * cc * (2.0 - cc) * covariance

                # Conservative step-size update.  Clipping the exponent avoids
                # numerical explosions on pathological objective values.
                exponent = (cs / damping) * (ps_norm / chi_n - 1.0)
                sigma *= math.exp(float(np.clip(exponent, -0.5, 0.5)))
                sigma = float(np.clip(sigma, 1e-9 * mean_span, 0.8 * mean_span))

                generation += 1
                if generation % 20 == 0:
                    covariance = 0.5 * (covariance + covariance.T)
                    try:
                        eigenvalues, basis = np.linalg.eigh(covariance)
                    except np.linalg.LinAlgError:
                        break
                    eigenvalues = np.maximum(eigenvalues, 1e-20)
                    scales = np.sqrt(eigenvalues)
                    inv_sqrt_cov = (basis / scales) @ basis.T

                if self._best_y < local_best - 1e-10 * max(1.0, abs(local_best)):
                    local_best = self._best_y
                    no_improve = 0
                else:
                    no_improve += 1

            restart += 1

    def _restart_mean(self, restart: int) -> np.ndarray:
        """Pick a restart center from the elite archive, with mild jitter."""
        if self._best_x is None:
            return self._lower + np.random.random(self.dim) * self._span

        self._archive.sort(key=lambda item: item[0])
        if restart == 0 or len(self._archive) < 2:
            mean = self._best_x.copy()
        else:
            # Most restarts exploit a good archived point; some explore from a
            # fresh random point so a bad early basin does not dominate forever.
            if np.random.random() < 0.80:
                rank = np.random.randint(min(8, len(self._archive)))
                mean = self._archive[rank][1].copy()
                jitter = np.random.normal(0.0, 0.12 * self._span, size=self.dim)
                mean = mean + jitter
            else:
                mean = self._lower + np.random.random(self.dim) * self._span

        return np.clip(mean, self._lower, self._upper)

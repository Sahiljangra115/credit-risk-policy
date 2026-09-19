"""Simulated A/B test of a new approval policy: sample size, CUPED, empirical power."""
import numpy as np
from scipy import stats
from . import config as C


def n_per_arm(delta, sd, alpha=C.ALPHA, power=C.POWER):
    """Two-sided two-sample z-test sample size per arm for mean difference delta."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * (z * sd / delta) ** 2))


def cuped(y, x):
    """Return (adjusted y, theta, variance reduction share). x is a pre-treatment covariate."""
    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    adj = y - theta * (x - x.mean())
    return adj, float(theta), float(1 - adj.var() / y.var())


def simulate_power(profit_a, profit_b, x, n, reps=1000, use_cuped=False, alpha=C.ALPHA, seed=C.SEED):
    """Each rep bootstraps 2n applicants, randomises into arms, runs Welch t-test.
    profit_a / profit_b are per-applicant outcomes under each policy; x is a pre-decision covariate."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(reps):
        idx = rng.choice(len(x), 2 * n, replace=True)  # bootstrap from the empirical test population
        arm = rng.permutation(2 * n) < n
        y = np.where(arm, profit_b[idx], profit_a[idx])
        if use_cuped:
            y, _, _ = cuped(y, x[idx])
        if stats.ttest_ind(y[arm], y[~arm], equal_var=False).pvalue < alpha:
            hits += 1
    return hits / reps

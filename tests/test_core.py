import numpy as np
import pandas as pd
import pytest
from src import ab, metrics, policy, reject_inference as ri
from src.models import expanding_folds, time_split
from src.woe import WoE, Scorecard

rng = np.random.default_rng(0)


def test_ks_perfect_and_random():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert metrics.ks(y, np.array([.1, .2, .3, .7, .8, .9])) == pytest.approx(1.0)
    y = rng.integers(0, 2, 20000)
    assert metrics.ks(y, rng.random(20000)) < 0.03


def test_ece_calibrated_vs_not():
    p = rng.random(50000)
    y = (rng.random(50000) < p).astype(int)
    assert metrics.ece(y, p) < 0.02
    assert metrics.ece(y, np.clip(p * 0.5, 0, 1)) > 0.1


def test_psi_zero_and_shift():
    a = rng.normal(size=20000)
    assert metrics.psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert metrics.psi(a, rng.normal(size=20000)) < 0.02
    assert metrics.psi(a, rng.normal(1.0, 1, 20000)) > 0.25


def test_breakeven_and_profit_sign():
    be = policy.breakeven_pd(0.15, 0.60)
    assert be == pytest.approx(0.2)
    assert policy.loan_profit(be, 100.0, 0.15, 0.60) == pytest.approx(0.0)
    assert policy.loan_profit(0.05, 100.0, 0.15, 0.60) > 0
    assert policy.loan_profit(0.40, 100.0, 0.15, 0.60) < 0


def test_realized_profit_known():
    y = np.array([0, 1, 0])
    e = np.array([100.0, 100.0, 200.0])
    assert policy.realized_profit(y, e, np.array([True, True, True]), 0.1, 0.5) == pytest.approx(10 - 50 + 20)
    assert policy.realized_profit(y, e, np.array([False, False, False])) == 0.0


def test_best_threshold_matches_breakeven_for_calibrated_scores():
    p = rng.random(200000) * 0.6
    cut = policy.best_threshold(p, np.ones_like(p))
    assert cut == pytest.approx(policy.breakeven_pd(), abs=0.01)


def test_best_threshold_rejects_all_when_unprofitable():
    p = np.full(100, 0.9)
    assert policy.best_threshold(p, np.ones(100)) == -np.inf


def test_woe_direction_nan_and_unseen():
    x = pd.Series(["a"] * 500 + ["b"] * 500 + [None] * 200 + ["c"] * 5)
    y = np.array([0] * 450 + [1] * 50 + [0] * 250 + [1] * 250 + [0] * 100 + [1] * 100 + [0, 1, 0, 1, 0])
    w = WoE().fit(x, y)
    assert w.woe["a"] > w.woe["b"]              # safer level has higher WoE
    assert w.iv > 0.1
    out = w.transform(pd.Series(["a", "b", "never_seen", None]))
    assert out[0] > out[1] and out[2] == 0.0
    assert np.isfinite(out).all()


def test_scorecard_learns_signal_and_drops_noise():
    n = 6000
    s = rng.normal(size=n)
    X = pd.DataFrame({"signal": s, "noise": rng.normal(size=n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-(s * 1.5 - 2)))).astype(int)
    sc = Scorecard().fit(X, y)
    assert "signal" in sc.enc and "noise" not in sc.enc
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, sc.predict_proba(X)) > 0.75


def test_time_split_is_ordered_and_disjoint():
    df = pd.DataFrame({"SK_ID_CURR": np.arange(1000), "app_month": [f"2020-{(i // 100) + 1:02d}" for i in range(1000)]})
    df = df.sample(frac=1, random_state=1)
    tr, ca, te = time_split(df)
    assert len(tr) + len(ca) + len(te) == 1000
    assert tr.app_month.max() <= ca.app_month.min() and ca.app_month.max() <= te.app_month.min()


def test_expanding_folds_never_leak():
    for tr, va in expanding_folds(1000, k=4):
        assert tr.start == 0 and tr.stop == va.start and va.stop > va.start


def test_legacy_rule_rejects_requested_share():
    df = pd.DataFrame({"credit_income_ratio": rng.random(5000), "annuity_income_ratio": rng.random(5000), "age_years": rng.uniform(20, 70, 5000)})
    acc, cut = ri.legacy_accept(df, reject_share=0.3)
    assert (1 - acc.mean()) == pytest.approx(0.3, abs=0.01)
    acc2, _ = ri.legacy_accept(df, cutoff=cut)
    assert (acc == acc2).all()


def test_parcel_augment_shapes_and_risk_multiplier():
    Xa = pd.DataFrame({"f": np.arange(10.0)}); Xr = pd.DataFrame({"f": np.arange(2000.0)})
    X, y = ri.parcel_augment(Xa, np.zeros(10), Xr, np.full(2000, 0.1), risk_mult=2.0)
    assert len(X) == len(y) == 2010
    assert y[10:].mean() == pytest.approx(0.2, abs=0.04)


def test_sample_size_formula_known_value():
    assert ab.n_per_arm(1.0, 1.0, 0.05, 0.8) == 16   # 2*(1.96+0.8416)^2 = 15.7
    assert ab.n_per_arm(0.5, 1.0) == pytest.approx(4 * 15.7, abs=4)


def test_cuped_reduces_variance_without_moving_mean():
    x = rng.normal(size=20000)
    y = 2 * x + rng.normal(size=20000)
    adj, theta, vr = ab.cuped(y, x)
    assert theta == pytest.approx(2.0, abs=0.1)
    assert vr == pytest.approx(0.8, abs=0.03)
    assert adj.mean() == pytest.approx(y.mean(), abs=1e-9)


def test_power_near_nominal_under_null_and_high_under_effect():
    x = rng.random(4000)
    a = rng.normal(size=4000); b = a.copy()
    assert ab.simulate_power(a, b, x, 300, reps=200) < 0.12
    assert ab.simulate_power(a, b + 0.5, x, 300, reps=200) > 0.9

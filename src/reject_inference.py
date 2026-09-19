"""Legacy policy simulation and reject inference.
Home Credit only has outcomes for loans that were granted. We cannot observe real rejects, so we SIMULATE a legacy
rule on the through-the-door population (whose outcomes we do know), hide the labels of the rejected share, and
measure how much a model trained on accepted-only data suffers, and how much parceling augmentation recovers."""
import numpy as np
import pandas as pd
from . import config as C


def legacy_score(df):
    """Crude risk rule: high credit-to-income and young age look risky. Higher = riskier."""
    r = lambda s: s.rank(pct=True).fillna(0.5)
    return (r(df["credit_income_ratio"]) + r(df["annuity_income_ratio"]) - r(df["age_years"])).values


def legacy_accept(df, cutoff=None, reject_share=C.REJECT_SHARE):
    """Boolean accept mask. Pass a cutoff learned on the training window to apply the same rule later."""
    s = legacy_score(df)
    if cutoff is None:
        cutoff = np.quantile(s, 1 - reject_share)
    return s <= cutoff, float(cutoff)


def parcel_augment(X_acc, y_acc, X_rej, pd_rej, n_parcels=10, risk_mult=1.5, seed=C.SEED):
    """Parceling: bucket rejects by model PD, draw labels with rate = min(1, risk_mult * mean PD of the bucket).
    risk_mult > 1 encodes the belief that rejects are worse than the accepted model thinks (an assumption)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(pd_rej)
    y_rej = np.zeros(len(pd_rej), dtype=int)
    for part in np.array_split(order, n_parcels):
        rate = min(1.0, risk_mult * pd_rej[part].mean())
        y_rej[part] = (rng.random(len(part)) < rate).astype(int)
    X = pd.concat([X_acc, X_rej], ignore_index=True)
    y = np.concatenate([np.asarray(y_acc), y_rej])
    return X, y

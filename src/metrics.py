import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss


def ks(y, p):
    o = np.argsort(p)
    y = np.asarray(y)[o]
    cg = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    cb = np.cumsum(y) / max(y.sum(), 1)
    return float(np.max(np.abs(cg - cb)))


def ece(y, p, bins=10):
    """Expected calibration error with equal-count bins."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.array_split(np.argsort(p), bins)
    return float(sum(len(i) / len(p) * abs(y[i].mean() - p[i].mean()) for i in idx))


def summary(y, p):
    auc = roc_auc_score(y, p)
    return {"auc": auc, "gini": 2 * auc - 1, "ks": ks(y, p), "ece": ece(y, p), "brier": brier_score_loss(y, p)}


def psi(expected, actual, bins=10):
    """Population stability index; bin edges are expected-sample quantiles."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

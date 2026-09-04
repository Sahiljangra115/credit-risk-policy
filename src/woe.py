"""WoE binning and logistic scorecard. Numeric: quantile bins, NaN gets its own bin. Categorical: levels."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class WoE:
    def __init__(self, n_bins=8, min_share=0.02, smooth=0.5):
        self.n_bins, self.min_share, self.smooth = n_bins, min_share, smooth

    def fit(self, x, y):
        y = np.asarray(y)
        self.numeric = pd.api.types.is_numeric_dtype(x)
        if self.numeric:
            q = np.unique(np.nanquantile(x, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
            self.edges = q
        keys = self._key(x)
        df = pd.DataFrame({"k": keys, "y": y})
        g = df.groupby("k")["y"].agg(["sum", "count"])
        # fold rare levels into "__rare__"
        rare = g.index[g["count"] < self.min_share * len(df)]
        self.rare = set(rare) if not self.numeric else set()
        df["k"] = df["k"].where(~df["k"].isin(self.rare), "__rare__")
        g = df.groupby("k")["y"].agg(["sum", "count"])
        bad = g["sum"] + self.smooth
        good = g["count"] - g["sum"] + self.smooth
        self.woe = np.log((good / good.sum()) / (bad / bad.sum()))
        self.iv = float(((good / good.sum() - bad / bad.sum()) * self.woe).sum())
        return self

    def _key(self, x):
        if self.numeric:
            arr = np.asarray(x, dtype=float)
            k = np.where(np.isnan(arr), "nan", np.searchsorted(self.edges, arr).astype(str))
            return pd.Series(k, index=x.index)
        return x.astype(object).where(x.notna(), "nan").astype(str)

    def transform(self, x):
        k = self._key(x)
        k = k.where(~k.isin(self.rare), "__rare__")
        return k.map(self.woe).fillna(0.0).astype(float).values


class Scorecard:
    """WoE transform of features with IV >= min_iv, then L2 logistic regression."""

    def __init__(self, min_iv=0.02, max_feats=40, C=0.5):
        self.min_iv, self.max_feats, self.C = min_iv, max_feats, C

    def fit(self, X, y):
        enc = {c: WoE().fit(X[c], y) for c in X.columns}
        keep = sorted((c for c in enc if enc[c].iv >= self.min_iv), key=lambda c: -enc[c].iv)[: self.max_feats]
        self.enc = {c: enc[c] for c in keep}
        self.iv = {c: enc[c].iv for c in keep}
        T = self._t(X)
        self.lr = LogisticRegression(C=self.C, max_iter=1000).fit(T, y)
        self.mean_woe = T.mean(0)
        return self

    def contributions(self, X):
        """Per-feature log-odds contribution vs the average applicant (classic scorecard reason codes).
        Columns follow X.columns; features dropped by the IV filter get 0."""
        part = (self._t(X) - self.mean_woe) * self.lr.coef_[0]
        out = np.zeros((len(X), len(X.columns)))
        for j, c in enumerate(self.enc):
            out[:, X.columns.get_loc(c)] = part[:, j]
        return out

    def _t(self, X):
        return np.column_stack([e.transform(X[c]) for c, e in self.enc.items()])

    def predict_proba(self, X):
        return self.lr.predict_proba(self._t(X))[:, 1]

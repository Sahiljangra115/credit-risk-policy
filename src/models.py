"""Time-aware splits, model fitting (WoE scorecard, LightGBM, CatBoost), random search, calibration, SHAP contributions."""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from . import config as C

ID_COLS = ["SK_ID_CURR", "TARGET", "app_month", "application_date"]
# CODE_GENDER is kept out of the models (fair lending); it is used only for fairness slicing.
EXCLUDED = ID_COLS + ["CODE_GENDER"]


def feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDED]


def time_split(df, train=0.60, calib=0.15):
    """Order by application date; first 60% train, next 15% calibration, last 25% test."""
    d = df.sort_values(["app_month", "SK_ID_CURR"], kind="stable").reset_index(drop=True)
    i, j = int(len(d) * train), int(len(d) * (train + calib))
    return d.iloc[:i].copy(), d.iloc[i:j].copy(), d.iloc[j:].copy()


def expanding_folds(n, k=4, first=0.4):
    """Expanding-window folds over time-ordered rows: train on [0, a), validate on [a, b)."""
    edges = np.linspace(first, 1.0, k + 1)
    return [(slice(0, int(n * a)), slice(int(n * a), int(n * b))) for a, b in zip(edges[:-1], edges[1:])]


def _lgb_frame(X, cats=None):
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) in ("string", "str"):
            X[c] = pd.Categorical(X[c]) if cats is None else pd.Categorical(X[c], categories=cats[c])
    return X


class GBM:
    def __init__(self, **kw):
        import lightgbm as lgb
        p = dict(n_estimators=300, learning_rate=0.04, num_leaves=31, min_child_samples=100, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, random_state=C.SEED, verbose=-1, n_jobs=4)
        p.update(kw)
        self.m = lgb.LGBMClassifier(**p)

    def fit(self, X, y, sample_weight=None):
        X = _lgb_frame(X)
        self.cats = {c: list(X[c].cat.categories) for c in X.columns if str(X[c].dtype) == "category"}
        self.m.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(_lgb_frame(X, self.cats))[:, 1]

    def contributions(self, X):
        """Native TreeSHAP values (log-odds), shape (n, n_features); last column of lightgbm output (bias) dropped."""
        return self.m.predict(_lgb_frame(X, self.cats), pred_contrib=True)[:, :-1]


class CatBoost:
    """CatBoost with native categorical handling. Missing categories become the string "nan"."""

    def __init__(self, **kw):
        from catboost import CatBoostClassifier
        p = dict(iterations=600, learning_rate=0.06, depth=6, l2_leaf_reg=5.0, random_seed=C.SEED, verbose=0,
                 thread_count=4, allow_writing_files=False)
        p.update(kw)
        self.m = CatBoostClassifier(**p)

    def _pool(self, X, y=None):
        from catboost import Pool
        X = X.copy()
        for c in self.cat_cols:
            X[c] = X[c].astype(object).where(X[c].notna(), "nan").astype(str)
        return Pool(X, y, cat_features=self.cat_cols)

    def fit(self, X, y, sample_weight=None):
        self.cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("string", "str", "category")]
        self.m.fit(self._pool(X, y), sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(self._pool(X))[:, 1]

    def contributions(self, X):
        """Native CatBoost SHAP values (log-odds), bias column dropped."""
        return self.m.get_feature_importance(self._pool(X), type="ShapValues")[:, :-1]


LGBM_SPACE = {"num_leaves": [15, 31, 63, 127], "learning_rate": [0.02, 0.03, 0.05, 0.08],
              "n_estimators": [200, 300, 500, 800], "min_child_samples": [50, 100, 200, 400],
              "colsample_bytree": [0.4, 0.5, 0.7, 0.9], "reg_lambda": [0.0, 1.0, 5.0, 20.0]}


def random_search_lgbm(X, y, n_trials=8, val_share=0.2, seed=C.SEED):
    """Random search on a time-ordered window: fit on the first (1 - val_share), score AUC on the rest.
    Callers pass the TRAIN window only, so neither the calibration nor the test window is used for tuning.
    Trial 0 is the default parameter set, so the search can only keep or beat it on this split."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    cut = int(len(X) * (1 - val_share))
    trials = [{}] + [{k: v[rng.integers(len(v))] for k, v in LGBM_SPACE.items()} for _ in range(n_trials - 1)]
    rows = []
    for i, params in enumerate(trials):
        m = GBM(**params).fit(X.iloc[:cut], y.iloc[:cut])
        rows.append({"trial": i, **params, "val_auc": roc_auc_score(y.iloc[cut:], m.predict_proba(X.iloc[cut:]))})
    res = pd.DataFrame(rows)
    best = trials[int(res.val_auc.idxmax())]
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}, res


class Platt:
    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._z(p), y)
        return self

    @staticmethod
    def _z(p):
        p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def predict(self, p):
        return self.lr.predict_proba(self._z(p))[:, 1]


class Isotonic:
    def fit(self, p, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        return self

    def predict(self, p):
        return self.iso.predict(p)

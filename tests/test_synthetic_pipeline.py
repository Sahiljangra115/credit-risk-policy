"""End to end on a small SYNTHETIC set: generate 7 tables -> committed SQL -> model table -> split -> GBM ->
calibration -> cost cutoff -> scoring demo. Checks plumbing only; the numbers are not claims."""
import duckdb
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from src import policy, score as S
from src.data import TABLES, _find
from src.features import model_table
from src.models import GBM, Isotonic, feature_cols, time_split
from src.synth import generate


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    out = tmp_path_factory.mktemp("syn")
    generate(n=4000, out=out)
    con = duckdb.connect()
    for v, names in TABLES.items():
        con.execute(f"CREATE VIEW {v} AS SELECT * FROM {_find(out, names)}")
    df = model_table(con)
    feats = feature_cols(df)
    tr, ca, te = time_split(df)
    m = GBM(n_estimators=80).fit(tr[feats], tr.TARGET)
    iso = Isotonic().fit(m.predict_proba(ca[feats]), ca.TARGET)
    cut = policy.best_threshold(iso.predict(m.predict_proba(ca[feats])), ca.AMT_CREDIT.values)
    b = {"model_name": "lgbm+isotonic", "model": m, "calibrator": iso, "feats": feats, "cutoff": cut,
         "margin": 0.15, "lgd": 0.6, "source": "SYNTHETIC", "test_ids": te.SK_ID_CURR.values}
    return df, feats, te, m, b


def test_model_table_shape_and_signal(pipe):
    df, feats, te, m, _ = pipe
    assert len(df) == 4000 and df.SK_ID_CURR.is_unique and len(feats) == 59
    assert roc_auc_score(te.TARGET, m.predict_proba(te[feats])) > 0.6   # synthetic data has a planted signal


def test_score_existing_row(pipe):
    _, feats, te, _, b = pipe
    out = S.score(S._frame(te.iloc[[0]], b), b)
    assert 0 < out["pd"] < 1 and out["decision"] in ("approve", "decline")
    assert len(out["top_risk_reasons"]) <= 4
    assert all(r["log_odds_impact"] > 0 for r in out["top_risk_reasons"])


def test_score_from_raw_fields_matches_sql(pipe):
    *_, b = pipe
    X = S.from_raw({"AMT_CREDIT": 900000, "AMT_INCOME_TOTAL": 100000, "AMT_ANNUITY": 50000, "DAYS_BIRTH": -9000,
                    "EXT_SOURCE_2": 0.1, "NAME_INCOME_TYPE": "Working"}, b)
    assert list(X.columns) == b["feats"] and len(X) == 1
    assert X.credit_income_ratio.iloc[0] == pytest.approx(9.0)
    assert np.isnan(X.bureau_n.iloc[0])                                    # no history given -> missing
    risky = S.score(X, b)
    safe = S.score(S.from_raw({"AMT_CREDIT": 100000, "AMT_INCOME_TOTAL": 300000, "AMT_ANNUITY": 5000,
                               "DAYS_BIRTH": -18000, "EXT_SOURCE_2": 0.9}, b), b)
    assert risky["pd"] > safe["pd"]

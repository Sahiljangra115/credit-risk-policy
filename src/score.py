"""Score one applicant: calibrated PD, decision under the cost policy, and the top reasons that raised the risk
(adverse-action style, from the model's SHAP values). Needs outputs/final_model.joblib from `python -m src.run`.

  python -m src.score 456789                                  # SK_ID_CURR from the test window
  python -m src.score --raw '{"AMT_CREDIT": 900000, "AMT_INCOME_TOTAL": 120000, "AMT_ANNUITY": 45000,
                              "DAYS_BIRTH": -9000, "EXT_SOURCE_2": 0.2}'
Raw mode runs the committed sql/01_application.sql on the one row. Fields left out (including all bureau and
previous-loan history) are treated as missing, which the model reads like an applicant with no record.
"""
import argparse
import json
import duckdb
import joblib
import numpy as np
import pandas as pd
from . import config as C
from .policy import loan_profit

BUNDLE = C.OUT / "final_model.joblib"

# Plain-language reason text for the features a credit officer will see most often. Others fall back to the name.
REASONS = {
    "ext_source_mean": "Low average external credit bureau score",
    "EXT_SOURCE_1": "Low external credit score (source 1)",
    "EXT_SOURCE_2": "Low external credit score (source 2)",
    "EXT_SOURCE_3": "Low external credit score (source 3)",
    "credit_income_ratio": "Loan amount is high relative to income",
    "annuity_income_ratio": "Repayment is high relative to income",
    "annuity_credit_ratio": "Repayment schedule (instalment vs loan size)",
    "credit_goods_ratio": "Loan amount is high relative to the goods price",
    "AMT_CREDIT": "Loan amount",
    "AMT_ANNUITY": "Instalment amount",
    "AMT_INCOME_TOTAL": "Income level",
    "age_years": "Applicant age",
    "employed_years": "Short time in current employment",
    "employed_flag_missing": "No current employment record",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_INCOME_TYPE": "Income type",
    "OCCUPATION_TYPE": "Occupation",
    "NAME_CONTRACT_TYPE": "Product type",
    "NAME_FAMILY_STATUS": "Family status",
    "FLAG_OWN_CAR": "Does not own a car",
    "REGION_RATING_CLIENT": "Area risk rating",
    "bureau_n": "Number of loans at other lenders",
    "bureau_active_n": "Number of active loans at other lenders",
    "bureau_debt_ratio": "High outstanding debt at other lenders",
    "bureau_debt_sum": "Total outstanding debt at other lenders",
    "bureau_max_day_overdue": "Overdue payment at another lender",
    "bureau_bb_late_share": "Late payments on loans at other lenders",
    "prev_refused_rate": "Previous applications were refused",
    "prev_refused_n": "Previous applications were refused",
    "prev_last_decision_days": "Recent previous application",
    "prev_credit_app_ratio": "Credit granted vs amount requested on previous applications",
    "prev_amt_application_mean": "Size of previous applications",
    "pos_dpd_max": "Days past due on previous point-of-sale loans",
    "cc_dpd_max": "Days past due on credit card",
    "inst_days_late_mean": "Late instalment payments on previous loans",
    "inst_days_late_max": "Late instalment payments on previous loans",
    "inst_late_share": "Late instalment payments on previous loans",
    "inst_late_share_12m": "Late instalment payments in the last 12 months",
    "inst_pay_ratio_mean": "Paid less than the instalment due",
    "inst_underpaid_n": "Paid less than the instalment due",
    "pos_dpd_share": "Days past due on previous point-of-sale loans",
    "pos_cnt_instalment_mean": "Long instalment terms on previous loans",
    "cc_utilisation_mean": "High credit card utilisation",
    "cc_utilisation_max": "High credit card utilisation",
    "cc_drawings_mean": "High credit card cash drawings",
}


def load_bundle(path=BUNDLE):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.run` first.")
    return joblib.load(path)


def _frame(row: pd.DataFrame, b) -> pd.DataFrame:
    """Order columns like training; categorical columns must be object dtype even when all missing."""
    X = row.reindex(columns=b["feats"])
    m = b["model"]
    for c in list(getattr(m, "cats", {})) + list(getattr(m, "cat_cols", [])):  # LightGBM / CatBoost
        X[c] = X[c].astype(object)
    return X


def from_raw(fields: dict, b) -> pd.DataFrame:
    """One raw application record -> one model-table row, using the committed application SQL."""
    raw_cols = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED",
                "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    app = pd.DataFrame([{**{c: np.nan for c in raw_cols}, **fields, "SK_ID_CURR": 0}])
    app[raw_cols] = app[raw_cols].astype(float)
    con = duckdb.connect()
    con.register("application", app)
    derived = con.execute((C.SQL / "01_application.sql").read_text().strip().rstrip(";")).df()
    row = app.drop(columns=[c for c in derived.columns if c != "SK_ID_CURR" and c in app.columns]).merge(derived, on="SK_ID_CURR")
    return _frame(row, b)


def from_id(sk_id: int, b) -> tuple[pd.DataFrame, int]:
    """Look up an applicant in the cached model table. Only test-window ids are allowed (the model never saw them)."""
    if sk_id not in set(b["test_ids"].tolist()):
        raise KeyError(f"SK_ID_CURR {sk_id} is not in the test window")
    cache = C.INTERIM / f"model_table_{b['source'].lower()}.parquet"
    row = duckdb.execute(f"SELECT * FROM read_parquet('{cache}') WHERE SK_ID_CURR = ?", [sk_id]).df()
    return _frame(row, b), int(row.TARGET.iloc[0])


def score(X: pd.DataFrame, b, n_reasons=4) -> dict:
    p = float(b["calibrator"].predict(b["model"].predict_proba(X))[0])
    contrib = b["model"].contributions(X)[0]
    order = [j for j in np.argsort(-contrib) if contrib[j] > 0][:n_reasons]
    reasons = []
    for j in order:
        f = b["feats"][j]
        v = X.iloc[0, j]
        reasons.append({"feature": f, "value": None if pd.isna(v) else (v if isinstance(v, str) else round(float(v), 4)),
                        "reason": f"No record for {f}" if pd.isna(v) else REASONS.get(f, f),
                        "log_odds_impact": round(float(contrib[j]), 4)})
    exposure = X["AMT_CREDIT"].iloc[0] if "AMT_CREDIT" in X else np.nan
    return {"pd": round(p, 4), "decision": "approve" if p <= b["cutoff"] else "decline",
            "pd_cutoff": round(b["cutoff"], 4), "model": b["model_name"],
            "expected_profit_if_approved": None if pd.isna(exposure) else round(float(loan_profit(p, exposure, b["margin"], b["lgd"])), 2),
            "top_risk_reasons": reasons}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sk_id_curr", nargs="?", type=int)
    ap.add_argument("--raw", help="JSON object of raw application fields")
    a = ap.parse_args()
    b = load_bundle()
    if a.raw:
        out = score(from_raw(json.loads(a.raw), b), b)
    elif a.sk_id_curr is not None:
        try:
            X, y = from_id(a.sk_id_curr, b)
        except KeyError as e:
            ap.error(e.args[0])
        out = {"sk_id_curr": a.sk_id_curr, **score(X, b), "observed_default": y}
    else:
        ap.error("give an SK_ID_CURR or --raw")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

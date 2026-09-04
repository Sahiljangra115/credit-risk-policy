"""End-to-end experiment: python -m src.run [--synthetic]. Writes outputs/ (outputs/synthetic/ for fake data).
Every number in README comes from here."""
import json
import os
import platform
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from . import config as C
from . import ab, metrics, policy, reject_inference as ri
from .data import connect
from .features import model_table
from .models import GBM, CatBoost, Isotonic, Platt, expanding_folds, feature_cols, random_search_lgbm, time_split
from .woe import Scorecard

R = {}  # results dict, dumped to outputs/metrics.json


def load(force_synth):
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    con, source = connect(force_synth)
    cache = C.INTERIM / f"model_table_{source.lower()}.parquet"
    if not cache.exists():
        df = model_table(con)
        con.register("mt", df)
        con.execute(f"COPY mt TO '{cache}' (FORMAT PARQUET)")
    import duckdb
    df = duckdb.sql(f"SELECT * FROM read_parquet('{cache}')").df()
    return df, source


def cv_compare(dev, feats, makers):
    rows = []
    for k, (tr, va) in enumerate(expanding_folds(len(dev))):
        a, b = dev.iloc[tr], dev.iloc[va]
        row = {"fold": k, "n_train": len(a), "n_val": len(b)}
        for name, make in makers.items():
            row[f"auc_{name}"] = roc_auc_score(b.TARGET, make().fit(a[feats], a.TARGET).predict_proba(b[feats]))
        rows.append(row)
        print(row, flush=True)
    return pd.DataFrame(rows)


def reliability_plot(y, curves, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 0.5], [0, 0.5], "k--", lw=1, label="perfect")
    for name, p in curves.items():
        idx = np.array_split(np.argsort(p), 10)
        ax.plot([p[i].mean() for i in idx], [np.asarray(y)[i].mean() for i in idx], "o-", ms=4, label=name)
    ax.set_xlabel("mean predicted PD (decile)"); ax.set_ylabel("observed default rate"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    t0 = time.time()
    synth = "--synthetic" in sys.argv
    out = C.OUT / "synthetic" if synth else C.OUT
    out.mkdir(parents=True, exist_ok=True)
    df, source = load(synth)
    R["data_source"] = source
    R["run_started"] = time.strftime("%Y-%m-%d %H:%M")
    R["hardware"] = f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical cores, CPU only, 4 threads per model"
    feats = feature_cols(df)
    R["n_rows"], R["n_features"], R["default_rate"] = len(df), len(feats), float(df.TARGET.mean())
    tr, ca, te = time_split(df)
    R["split_rows"] = {"train": len(tr), "calib": len(ca), "test": len(te)}
    R["split_months"] = {k: [d.app_month.min(), d.app_month.max()] for k, d in [("train", tr), ("calib", ca), ("test", te)]}
    dr = df.groupby("app_month").TARGET.agg(n="size", default_rate="mean").reset_index()
    dr.to_csv(out / "default_rate_by_month.csv", index=False)  # evidence for README caveat 1 (flat monthly default rate)
    print(source, R["n_rows"], "rows", len(feats), "features", flush=True)

    # 0. LightGBM random search on the TRAIN window only (last 20% of train is the validation slice)
    lgbm_params, search = random_search_lgbm(tr[feats], tr.TARGET, n_trials=8)
    search.to_csv(out / "lgbm_random_search.csv", index=False)
    R["lgbm_search"] = {"n_trials": len(search), "default_val_auc": float(search.val_auc.iloc[0]),
                        "best_val_auc": float(search.val_auc.max()), "best_params": lgbm_params or "default kept"}
    print(search.to_string(), flush=True)

    makers = {"scorecard": Scorecard, "lgbm": GBM, "catboost": CatBoost}
    if lgbm_params:
        makers["lgbm_tuned"] = lambda: GBM(**lgbm_params)

    # 1. time-aware CV on the development window (train + calib)
    dev = pd.concat([tr, ca], ignore_index=True)
    cv = cv_compare(dev, feats, makers)
    cv.to_csv(out / "cv_folds.csv", index=False)
    R["cv_auc_mean"] = {k: float(cv[f"auc_{k}"].mean()) for k in makers}
    R["cv_auc_std"] = {k: float(cv[f"auc_{k}"].std()) for k in makers}
    print(cv, flush=True)

    # 2. final fit on train, evaluate on the untouched test window
    fitted = {k: make().fit(tr[feats], tr.TARGET) for k, make in makers.items()}
    raw = {k: (m.predict_proba(ca[feats]), m.predict_proba(te[feats])) for k, m in fitted.items()}
    R["test_raw"] = {k: metrics.summary(te.TARGET, v[1]) for k, v in raw.items()}
    sc = fitted["scorecard"]
    R["scorecard_top_iv"] = dict(sorted(sc.iv.items(), key=lambda kv: -kv[1])[:10])
    R["scorecard_n_features"] = len(sc.iv)

    # 3. calibration, fitted on the calibration window only
    R["test_calibrated"] = {}
    cal, calibrators = {}, {}
    for name, (pc, pt) in raw.items():
        for cname, Cal in [("platt", Platt), ("isotonic", Isotonic)]:
            m = Cal().fit(pc, ca.TARGET)
            calibrators[(name, cname)] = m
            cal[(name, cname)] = (m.predict(pc), m.predict(pt))
            R["test_calibrated"][f"{name}+{cname}"] = metrics.summary(te.TARGET, m.predict(pt))

    # 5 (decided before looking at test): final model = best CV AUC + isotonic.
    best = max(R["cv_auc_mean"], key=R["cv_auc_mean"].get)
    R["final_model"] = best + "+isotonic"
    final = fitted[best]
    reliability_plot(te.TARGET.values, {f"{best} raw": raw[best][1], f"{best} isotonic": cal[(best, "isotonic")][1],
                                         "scorecard raw": raw["scorecard"][1]}, out / "reliability.png")

    # 4. SHAP (native TreeSHAP for the GBMs, WoE reason codes for the scorecard) on a test sample, final model
    samp = te.sample(min(5000, len(te)), random_state=C.SEED)
    sv = final.contributions(samp[feats])
    imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False)
    imp.to_csv(out / "shap_importance.csv", header=["mean_abs_shap_logodds"])
    R["shap_model"] = best
    R["shap_top10"] = {k: float(v) for k, v in imp.head(10).items()}
    fig, ax = plt.subplots(figsize=(6, 5)); imp.head(15)[::-1].plot.barh(ax=ax)
    ax.set_xlabel(f"mean |SHAP| (log-odds), {best}"); fig.tight_layout(); fig.savefig(out / "shap_top15.png", dpi=130); plt.close(fig)

    pdc, pdt = cal[(best, "isotonic")]
    exp_c, exp_t = ca.AMT_CREDIT.values, te.AMT_CREDIT.values
    yt = te.TARGET.values
    cut = policy.best_threshold(pdc, exp_c)
    R["policy"] = {"margin": C.MARGIN, "lgd": C.LGD, "breakeven_pd": policy.breakeven_pd(), "chosen_pd_cutoff": cut}
    ap_model = pdt <= cut
    _, leg_cut = ri.legacy_accept(tr)
    ap_leg, _ = ri.legacy_accept(te, leg_cut)
    ap_all = np.ones(len(te), bool)
    ap_same = pdt <= np.quantile(pdt, 1 - C.REJECT_SHARE)  # model rejects the same share the legacy rule does
    rows = []
    for name, ap in [("approve_all", ap_all), ("legacy_rule", ap_leg), ("model_reject_same_share", ap_same), ("model_cost_threshold", ap_model)]:
        rows.append({"policy": name, "approval_rate": ap.mean(), "default_rate_approved": yt[ap].mean(),
                     "realized_profit": policy.realized_profit(yt, exp_t, ap),
                     "expected_profit_model": float(policy.loan_profit(pdt[ap], exp_t[ap]).sum())})
    pol = pd.DataFrame(rows)
    pol["profit_vs_approve_all"] = pol.realized_profit - pol.loc[pol.policy == "approve_all", "realized_profit"].iloc[0]
    pol["profit_vs_legacy"] = pol.realized_profit - pol.loc[pol.policy == "legacy_rule", "realized_profit"].iloc[0]
    pol.to_csv(out / "policy_comparison.csv", index=False)
    R["policy_table"] = pol.to_dict("records")
    print(pol.to_string(), flush=True)

    # threshold sweep on test (diagnostic only; the chosen cutoff came from calib)
    qs = np.linspace(0.3, 1.0, 36)
    sweep = [(q, policy.realized_profit(yt, exp_t, pdt <= np.quantile(pdt, q)), (pdt <= np.quantile(pdt, q)).mean()) for q in qs]
    sw = pd.DataFrame(sweep, columns=["pd_quantile_cut", "realized_profit", "approval_rate"])
    sw.to_csv(out / "threshold_sweep.csv", index=False)
    R["hindsight_best_sweep"] = sw.sort_values("realized_profit").iloc[-1].to_dict()
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(sw.approval_rate, sw.realized_profit / 1e9)
    ax.axvline(ap_model.mean(), color="C1", ls="--", label="chosen on calib"); ax.set_xlabel("approval rate")
    ax.set_ylabel("realized profit on test (1e9, source currency)"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "threshold_sweep.png", dpi=130); plt.close(fig)

    # sensitivity to the economic assumptions (cutoff re-learned on calib for each pair)
    sens = []
    for mg in [0.05, 0.10, 0.15, 0.20]:
        for lg in [0.40, 0.60, 0.80]:
            cu = policy.best_threshold(pdc, exp_c, mg, lg)
            ap = pdt <= cu
            sens.append({"margin": mg, "lgd": lg, "breakeven_pd": policy.breakeven_pd(mg, lg), "approval_rate": float(ap.mean()),
                         "profit_vs_approve_all": policy.realized_profit(yt, exp_t, ap, mg, lg) - policy.realized_profit(yt, exp_t, ap_all, mg, lg)})
    pd.DataFrame(sens).to_csv(out / "policy_sensitivity.csv", index=False)

    # model bundle for the scoring demo (src/score.py). Trained on the train window only, as evaluated above.
    joblib.dump({"model_name": R["final_model"], "model": final, "calibrator": calibrators[(best, "isotonic")],
                 "feats": feats, "cutoff": cut, "margin": C.MARGIN, "lgd": C.LGD, "source": source,
                 "test_ids": te.SK_ID_CURR.values}, out / "final_model.joblib")

    # 6. reject inference experiment (LightGBM with the selected params, train window; test window is fully observed)
    GBMs = lambda: GBM(**lgbm_params)
    acc, lc = ri.legacy_accept(tr)
    trA, trR = tr[acc], tr[~acc]
    teA, _ = ri.legacy_accept(te, lc)
    m_all = GBMs().fit(tr[feats], tr.TARGET)
    m_acc = GBMs().fit(trA[feats], trA.TARGET)
    pr = m_acc.predict_proba(trR[feats])
    aug = {}
    for mult in (1.0, 1.5):
        Xa, ya = ri.parcel_augment(trA[feats], trA.TARGET.values, trR[feats], pr, risk_mult=mult)
        aug[mult] = GBMs().fit(Xa, ya)
    ri_rows = []
    for name, m in [("oracle_all_labels", m_all), ("accepted_only", m_acc), ("accepted+parceling_x1.0", aug[1.0]), ("accepted+parceling_x1.5", aug[1.5])]:
        p = m.predict_proba(te[feats])
        rej = ~teA
        ri_rows.append({"model": name, "auc_full_test": roc_auc_score(yt, p), "ks_full_test": metrics.ks(yt, p),
                        "auc_legacy_rejected_slice": roc_auc_score(yt[rej], p[rej]),
                        "mean_pd_on_rejected": p[rej].mean(), "actual_default_rate_rejected": yt[rej].mean(),
                        "mean_pd_on_accepted": p[~rej].mean(), "actual_default_rate_accepted": yt[~rej].mean()})
    rid = pd.DataFrame(ri_rows)
    rid.to_csv(out / "reject_inference.csv", index=False)
    R["reject_inference"] = rid.to_dict("records"); print(rid.T.to_string(), flush=True)

    # 7. fairness slices at the chosen cutoff (test window)
    fr = []
    for col in ["CODE_GENDER", "REGION_RATING_CLIENT"]:
        for lvl, g in te.assign(ap=ap_model, pd_=pdt).groupby(col):
            fr.append({"slice": col, "level": str(lvl), "n": len(g), "approval_rate": g.ap.mean(),
                       "default_rate_all": g.TARGET.mean(), "default_rate_approved": g.TARGET[g.ap].mean(),
                       "mean_pd": g.pd_.mean()})
    fa = pd.DataFrame(fr)
    fa["approval_ratio_vs_max"] = fa.groupby("slice").approval_rate.transform(lambda s: s / s.max())
    fa.to_csv(out / "fairness.csv", index=False); R["fairness"] = fa.to_dict("records")
    print(fa.to_string(), flush=True)

    # 8. PSI of the score by application month vs the training window
    ps = []
    base = calibrators[(best, "isotonic")].predict(final.predict_proba(tr[feats]))
    allp = pd.concat([ca, te], ignore_index=True)
    allp["score"] = np.concatenate([pdc, pdt])
    for mth, g in allp.groupby("app_month"):
        if len(g) >= 500:
            ps.append({"app_month": mth, "n": len(g), "psi": metrics.psi(base, g.score.values), "mean_score": g.score.mean(), "default_rate": g.TARGET.mean()})
    psi = pd.DataFrame(ps); psi.to_csv(out / "psi_by_month.csv", index=False)
    R["psi"] = {"max": float(psi.psi.max()) if len(psi) else None, "n_months": len(psi), "months_over_0.10": int((psi.psi > 0.10).sum()), "months_over_0.25": int((psi.psi > 0.25).sum())}
    if len(psi):
        fig, ax = plt.subplots(figsize=(7, 3.5)); ax.bar(psi.app_month, psi.psi); ax.axhline(0.1, color="orange", ls="--"); ax.axhline(0.25, color="red", ls="--")
        ax.set_ylabel("PSI vs train score"); plt.xticks(rotation=90, fontsize=6); fig.tight_layout(); fig.savefig(out / "psi_by_month.png", dpi=130); plt.close(fig)

    # 9. simulated A/B: baseline policy (A) vs model cost threshold (B), per-applicant realized profit.
    # Two baselines: approve_all (what the dataset's lender effectively did) and the simulated legacy rule.
    def per_app(ap):
        return np.where(ap, np.where(yt == 1, -C.LGD, C.MARGIN) * exp_t, 0.0)
    pb = per_app(ap_model)
    R["ab"] = {}
    for bname, ap_base in [("vs_approve_all", ap_all), ("vs_legacy_rule", ap_leg)]:
        pa = per_app(ap_base)
        delta = float(pb.mean() - pa.mean())
        sd = float(np.sqrt((pa.var(ddof=1) + pb.var(ddof=1)) / 2))
        r = {"mean_profit_per_applicant_A": float(pa.mean()), "mean_profit_per_applicant_B": float(pb.mean()), "delta": delta, "sd_per_applicant": sd}
        if delta > 0:
            rng = np.random.default_rng(C.SEED)
            arm = rng.random(len(pdt)) < 0.5
            _, theta, vr = ab.cuped(np.where(arm, pb, pa), pdt)
            n_raw, n_cup = ab.n_per_arm(delta, sd), ab.n_per_arm(delta, sd * np.sqrt(1 - vr))
            r.update({"n_per_arm_raw": n_raw, "n_per_arm_cuped": n_cup, "cuped_variance_reduction": vr, "cuped_theta": theta,
                      "weeks_raw": 2 * n_raw / C.WEEKLY_APPLICATIONS, "weeks_cuped": 2 * n_cup / C.WEEKLY_APPLICATIONS,
                      "empirical_power_raw_at_n_raw": ab.simulate_power(pa, pb, pdt, n_raw, reps=400),
                      "empirical_power_cuped_at_n_cuped": ab.simulate_power(pa, pb, pdt, n_cup, reps=400, use_cuped=True)})
        R["ab"][bname] = r
    print(R["ab"], flush=True)

    R["runtime_minutes"] = round((time.time() - t0) / 60, 1)
    (out / "metrics.json").write_text(json.dumps(R, indent=2, default=float))
    print("wrote", out / "metrics.json")


if __name__ == "__main__":
    main()

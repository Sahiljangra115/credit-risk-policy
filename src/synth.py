"""SYNTHETIC seven-table generator that mimics the Home Credit schema (columns the pipeline uses).
Everything it produces is fake. Metrics computed on it are plumbing checks, not claims."""
import numpy as np
import pandas as pd
from . import config as C

N_APP = 30000
N_MONTHS = 24


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _rep(rng, n_per, lo, size):
    return np.maximum(rng.poisson(n_per, size), lo)


def generate(n=N_APP, seed=C.SEED, out=C.SYN):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    ids = np.arange(100001, 100001 + n)
    z = rng.normal(size=n)  # latent riskiness, higher = riskier
    month = np.sort(rng.integers(0, N_MONTHS, n))  # application month index
    drift = 1 + 0.015 * month  # income inflation and mild mix shift over time
    income = np.exp(rng.normal(11.8, 0.5, n)) * drift
    credit = income * np.exp(rng.normal(1.0, 0.5, n))
    annuity = credit / rng.uniform(12, 60, n)
    age = rng.uniform(21, 65, n)
    gender = rng.choice(["F", "M"], n, p=[0.65, 0.35])
    region = rng.choice([1, 2, 3], n, p=[0.1, 0.7, 0.2])
    ext = lambda s: np.clip(0.5 - 0.18 * z + rng.normal(0, s, n), 0, 1)
    e1, e2, e3 = ext(0.2), ext(0.15), ext(0.2)
    e1[rng.random(n) < 0.55] = np.nan
    e3[rng.random(n) < 0.2] = np.nan
    employed = np.where(rng.random(n) < 0.18, 365243, -rng.uniform(100, 9000, n)).astype(int)
    lin = (-2.65 + 0.85 * z + 0.25 * np.log(credit / income) - 0.01 * (age - 40)
           + 0.1 * (region == 3) - 0.1 * (employed == 365243))
    target = (rng.random(n) < _sig(lin)).astype(int)
    app = pd.DataFrame({
        "SK_ID_CURR": ids, "TARGET": target,
        "application_date": pd.to_datetime("2022-01-01") + pd.to_timedelta(month * 30 + rng.integers(0, 30, n), "D"),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
        "CODE_GENDER": gender, "FLAG_OWN_CAR": rng.choice(["Y", "N"], n), "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n),
        "CNT_CHILDREN": rng.poisson(0.5, n), "CNT_FAM_MEMBERS": 1 + rng.poisson(1.2, n),
        "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": credit * rng.uniform(0.85, 1.0, n),
        "NAME_INCOME_TYPE": rng.choice(["Working", "Commercial associate", "Pensioner", "State servant"], n),
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary", "Higher education", "Incomplete higher"], n, p=[0.7, 0.25, 0.05]),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single", "Separated"], n, p=[0.6, 0.25, 0.15]),
        "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "With parents", "Rented apartment"], n, p=[0.85, 0.1, 0.05]),
        "OCCUPATION_TYPE": rng.choice(["Laborers", "Core staff", "Sales staff", "Managers", None], n),
        "REGION_RATING_CLIENT": region, "DAYS_BIRTH": (-age * 365.25).astype(int),
        "DAYS_EMPLOYED": employed, "EXT_SOURCE_1": e1, "EXT_SOURCE_2": e2, "EXT_SOURCE_3": e3,
    })
    app.to_parquet(out / "application_train.parquet")

    # bureau: ~3 loans/customer, riskier customers more overdue
    k = _rep(rng, 3, 0, n)
    cid = np.repeat(ids, k)
    zz = np.repeat(z, k)
    m = len(cid)
    bureau = pd.DataFrame({
        "SK_ID_CURR": cid, "SK_ID_BUREAU": np.arange(5000000, 5000000 + m),
        "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], m, p=[0.4, 0.6]),
        "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Car loan", "Mortgage"], m, p=[0.7, 0.2, 0.06, 0.04]),
        "AMT_CREDIT_SUM": np.exp(rng.normal(11.5, 1.0, m)),
        "AMT_CREDIT_SUM_DEBT": np.exp(rng.normal(10.5, 1.2, m)),
        "AMT_CREDIT_SUM_OVERDUE": np.where(rng.random(m) < _sig(-3 + 0.8 * zz), rng.exponential(2000, m), 0.0),
        "CREDIT_DAY_OVERDUE": np.where(rng.random(m) < _sig(-3 + 0.8 * zz), rng.integers(1, 90, m), 0),
    })
    bureau.to_parquet(out / "bureau.parquet")
    kb = _rep(rng, 6, 0, len(bureau))
    bid = np.repeat(bureau.SK_ID_BUREAU.values, kb)
    zb = np.repeat(zz, kb)
    p = _sig(-2.8 + 0.7 * zb)
    status = np.where(rng.random(len(bid)) < p, rng.choice(list("12345"), len(bid)), rng.choice(["0", "C", "X"], len(bid)))
    pd.DataFrame({"SK_ID_BUREAU": bid, "MONTHS_BALANCE": -rng.integers(0, 60, len(bid)), "STATUS": status}
                 ).to_parquet(out / "bureau_balance.parquet")

    # previous applications: riskier customers get refused more
    k = _rep(rng, 3, 0, n)
    cid = np.repeat(ids, k); zz = np.repeat(z, k); m = len(cid)
    refused = rng.random(m) < _sig(-1.6 + 0.6 * zz)
    amt = np.exp(rng.normal(11.0, 0.9, m))
    pd.DataFrame({
        "SK_ID_PREV": np.arange(2000000, 2000000 + m), "SK_ID_CURR": cid,
        "NAME_CONTRACT_STATUS": np.where(refused, "Refused", rng.choice(["Approved", "Canceled"], m, p=[0.9, 0.1])),
        "AMT_APPLICATION": amt, "AMT_CREDIT": np.where(refused, 0.0, amt * rng.uniform(0.9, 1.1, m)),
        "DAYS_DECISION": -rng.integers(30, 2500, m),
    }).to_parquet(out / "previous_application.parquet")
    prev = pd.read_parquet(out / "previous_application.parquet", columns=["SK_ID_PREV", "SK_ID_CURR"])
    ok = prev[prev.SK_ID_CURR.isin(ids)]
    zmap = pd.Series(z, index=ids)

    # installments
    k = _rep(rng, 8, 1, len(ok))
    pid = np.repeat(ok.SK_ID_PREV.values, k); cid = np.repeat(ok.SK_ID_CURR.values, k)
    zi = zmap.loc[cid].values; m = len(pid)
    due = -rng.integers(1, 1500, m)
    late = np.where(rng.random(m) < _sig(-2.0 + 0.8 * zi), rng.integers(1, 40, m), -rng.integers(0, 5, m))
    inst = rng.uniform(2000, 30000, m)
    short = rng.random(m) < _sig(-2.5 + 0.6 * zi)
    pd.DataFrame({"SK_ID_PREV": pid, "SK_ID_CURR": cid, "DAYS_INSTALMENT": due, "DAYS_ENTRY_PAYMENT": due + late,
                  "AMT_INSTALMENT": inst, "AMT_PAYMENT": np.where(short, inst * rng.uniform(0.3, 0.99, m), inst)}
                 ).to_parquet(out / "installments_payments.parquet")

    # POS cash
    k = _rep(rng, 6, 1, len(ok))
    pid = np.repeat(ok.SK_ID_PREV.values, k); cid = np.repeat(ok.SK_ID_CURR.values, k)
    zi = zmap.loc[cid].values; m = len(pid)
    pd.DataFrame({"SK_ID_PREV": pid, "SK_ID_CURR": cid, "MONTHS_BALANCE": -rng.integers(1, 80, m),
                  "CNT_INSTALMENT": rng.integers(6, 48, m).astype(float),
                  "NAME_CONTRACT_STATUS": rng.choice(["Active", "Completed"], m, p=[0.8, 0.2]),
                  "SK_DPD": np.where(rng.random(m) < _sig(-3.2 + 0.7 * zi), rng.integers(1, 60, m), 0)}
                 ).to_parquet(out / "pos_cash_balance.parquet")

    # credit card: 30% of customers
    has = rng.random(n) < 0.3
    k = np.where(has, _rep(rng, 10, 1, n), 0)
    cid = np.repeat(ids, k); zi = np.repeat(z, k); m = len(cid)
    limit = np.repeat(np.exp(rng.normal(11.5, 0.5, n)), k)
    pd.DataFrame({"SK_ID_PREV": np.repeat(np.arange(n) + 3000000, k), "SK_ID_CURR": cid,
                  "AMT_CREDIT_LIMIT_ACTUAL": limit,
                  "AMT_BALANCE": limit * np.clip(rng.beta(2, 4, m) + 0.05 * zi, 0, 1.2),
                  "AMT_DRAWINGS_CURRENT": rng.exponential(5000, m),
                  "SK_DPD": np.where(rng.random(m) < _sig(-3.5 + 0.7 * zi), rng.integers(1, 30, m), 0)}
                 ).to_parquet(out / "credit_card_balance.parquet")

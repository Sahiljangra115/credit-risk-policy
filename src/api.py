"""Scoring API on port 8030: uvicorn src.api:app --port 8030
GET  /score/{sk_id_curr}   applicant from the test window
POST /score                JSON object of raw application fields (see src/score.py)"""
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from . import score as S

app = FastAPI(title="credit-risk-policy scorer")


@lru_cache
def bundle():
    return S.load_bundle()


@app.get("/health")
def health():
    b = bundle()
    return {"model": b["model_name"], "pd_cutoff": b["cutoff"], "margin": b["margin"], "lgd": b["lgd"]}


@app.get("/score/{sk_id_curr}")
def score_id(sk_id_curr: int):
    try:
        X, y = S.from_id(sk_id_curr, bundle())
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"sk_id_curr": sk_id_curr, **S.score(X, bundle()), "observed_default": y}


@app.post("/score")
def score_raw(fields: dict):
    return S.score(S.from_raw(fields, bundle()), bundle())

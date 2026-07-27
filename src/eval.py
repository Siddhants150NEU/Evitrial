from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import traceback
import bm25s, ir_measures
import logging
from ir_measures import MAP, R, calc_aggregate, nDCG
from transformers import AutoModelForSequenceClassification
from . import match
from .schemas import Criterion
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from . import ingest
import bm25s, torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

from .config import loadConfig, setSeeds

RUNS_DIR = "reports/runs"

def _gitShortSha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"

def _newRunId() -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{_gitShortSha()}"

def _dump(runDir: str, name: str, obj) -> None:
    with open(os.path.join(runDir, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)

def _safe(fn, config: dict):
    try:
        return fn(config)
    except NotImplementedError:
        return {"status": "not_implemented"}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "trace": traceback.format_exc().splitlines()[-3:]}

def runEval(config: dict) -> str:
    runId = _newRunId()
    runDir = os.path.join(RUNS_DIR, runId)
    os.makedirs(runDir, exist_ok=True)

    _dump(runDir, "config.json", config)
    _dump(runDir, "meta.json", {
        "runId": runId,
        "gitShortSha": _gitShortSha(),
        "seed": config.get("seed"),
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    })

    metrics = {
        "retrieval": _safe(retrievalMetrics, config),
        "criterion": _safe(criterionMetrics, config),
        "faithfulness": _safe(faithfulnessMetrics, config),
        "abstention": _safe(abstentionMetrics, config),
        "calibration": _safe(calibration, config),
        "efficiency": _safe(latency, config),
    }
    _dump(runDir, "metrics.json", metrics)

    print(f"wrote {runDir}")
    for name, value in metrics.items():
        status = value.get("status", "ok") if isinstance(value, dict) else "ok"
        print(f"  {name:<14} {status}")
    return runId

def retrievalMetrics(config: dict) -> dict:

    rc = config["retrieval"]
    trials = list(ingest.loadTrials())
    nctIds = [t.nctId for t in trials]

    qrels: dict = {}                                  
    for qid, docid, rel in ingest.loadQrels():
        qrels.setdefault(qid, {})[docid] = int(rel)
    topics = list(ingest.loadTopics())                

    bm = bm25s.BM25()                                  
    bm.index(bm25s.tokenize([t.searchText() for t in trials], stopwords="en"))
    tok = AutoTokenizer.from_pretrained(rc["queryEncoder"])   
    model = AutoModel.from_pretrained(rc["queryEncoder"])
    client = QdrantClient(path=rc["qdrant"]["location"])

    def bm25Run():
        run = {}
        for qid, note in topics:
            idx, sc = bm.retrieve(bm25s.tokenize(note, stopwords="en"), k=rc["topN"])
            run[qid] = {nctIds[i]: float(s) for i, s in zip(idx[0], sc[0])}
        return run

    def denseRun():
        run = {}
        for qid, note in topics:
            enc = tok(note, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                vec = model(**enc)[0][:, 0][0].numpy().tolist()
            hits = client.query_points(collection_name=rc["qdrant"]["collection"],
                                       query=vec, limit=rc["topN"]).points
            run[qid] = {h.payload["nctId"]: float(h.score) for h in hits}
        return run

    def _norm(d):
        if not d: return {}
        lo, hi = min(d.values()), max(d.values())
        return {k: 1.0 for k in d} if hi == lo else {k: (v-lo)/(hi-lo) for k, v in d.items()}

    def fuse(ra, rb, alpha):
        out = {}
        for qid in ra:
            na, nb = _norm(ra[qid]), _norm(rb.get(qid, {}))
            out[qid] = {d: alpha*nb.get(d, 0) + (1-alpha)*na.get(d, 0) for d in set(na) | set(nb)}
        return out
    
    bm25R, denseR = bm25Run(), denseRun()
    hybridR = fuse(bm25R, denseR, rc["alpha"])


    ceTok = AutoTokenizer.from_pretrained(rc["crossEncoder"])
    ce = AutoModelForSequenceClassification.from_pretrained(rc["crossEncoder"])
    textById = {t.nctId: t.searchText() for t in trials}     # once
    noteById = dict(topics)

    def rerankRun(base, keep=50):
        out = {}
        for qid, hits in base.items():
            ids = sorted(hits, key=hits.get, reverse=True)[:keep] 
            pairs = [[noteById[qid], textById[i]] for i in ids]
            enc = ceTok(pairs,  truncation=True, max_length=512, padding=True, return_tensors="pt")
            with torch.no_grad():
                sc = ce(**enc).logits.squeeze(-1)
            out[qid] = {i: float(s) for i, s in zip(ids, sc)}
        return out
    
    rerankR = rerankRun(hybridR)

    measures = [nDCG@10, R@10, R@20, R@50, MAP]
    results = {}
    for name, run in [("bm25", bm25R), ("dense", denseR),
                      ("hybrid", hybridR), ("hybrid+rerank", rerankR)]:
        results[name] = {str(m): round(float(v), 4) for m, v in calc_aggregate(measures, qrels, run).items()}
    return results

    # measures = [nDCG@10, R@10, R@20, R@50, MAP]
    # results = {}
    # for name, run in [("bm25", bm25Run()), ("dense", denseRun())]:
    #     agg = calc_aggregate(measures, qrels, run)
    #     results[name] = {str(m): round(float(v), 4) for m, v in agg.items()}
    # return results

import numpy as np
def _boostrapCI(yTrue, yPred, labels, nresamples = 1000, seed = None):
    rng = np.random.default_rng(seed)
    yTrueArr, yPredArr = np.array(yTrue), np.array(yPred)
    n = len(yTrue)
    scores = []
    for a in range(nresamples):
        idx = rng.integers(0, n, size = n)
        scores.append(f1_score(yTrueArr[idx], yPredArr[idx], labels=labels, average="macro", zero_division=0))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)

# def criterionMetrics(config: dict) -> dict:
#     rows = ingest.loadAnnotations()
#     pairs = ingest.toEvalPairs(rows)
#     valPairs = ingest.splitPairs(pairs, config)["val"]

#     labels = ["MET", "NOT_MET", "UNKNOWN"]
#     yTrue, yPred = [], []
#     for pair in valPairs:
#         criterion = Criterion(
#             criterionId=pair.criterionId,
#             nctId=pair.nctId,
#             text=pair.criterionText,
#             criterionType=pair.criterionType,
#         )
#         decision = match.match(pair.note, criterion, config)
#         yTrue.append(pair.label)
#         yPred.append(decision.label)

#     macroF1 = f1_score(yTrue, yPred, labels=labels, average="macro", zero_division=0)
#     ciLow, ciHigh = _boostrapCI(yTrue, yPred, labels, nresamples=1000, seed=config["seed"])
#     precision, recall, f1, support = precision_recall_fscore_support(
#         yTrue, yPred, labels=labels, zero_division=0
#     )
#     perClass = {
#         lab: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
#         for lab, p, r, f, s in zip(labels, precision, recall, f1, support)
#     }
#     cm = confusion_matrix(yTrue, yPred, labels=labels).tolist()

#     return {
#         "rung": config["matcher"]["rung"],
#         "macroF1": float(macroF1),
#         "perClass": perClass,
#         "confusion": {"labels": labels, "matrix":cm},
#         "macroF1CI": [ciLow, ciHigh],
#         "n": len(valPairs),
#     }

def criterionMetrics(config: dict) -> dict:
    rows = ingest.loadAnnotations()
    pairs = ingest.toEvalPairs(rows)
    valPairs = ingest.splitPairs(pairs, config)["val"]
    labels = ["MET", "NOT_MET", "UNKNOWN"]

    results = {}
    for rung in ["rules", "zeroShot", "lora"]:
        rungConfig = {**config, "matcher": {**config["matcher"], "rung": rung}}
        yTrue, yPred = [], []
        try:
            for pair in valPairs:
                criterion = Criterion(
                    criterionId=pair.criterionId,
                    nctId=pair.nctId,
                    text=pair.criterionText,
                    criterionType=pair.criterionType,
                )
                decision = match.match(pair.note, criterion, rungConfig)
                yTrue.append(pair.label)
                yPred.append(decision.label)
        except NotImplementedError:
            results[rung] = {"status": "not_implemented"}
            continue

        macroF1 = f1_score(yTrue, yPred, labels=labels, average="macro", zero_division=0)
        ciLow, ciHigh = _boostrapCI(yTrue, yPred, labels, nresamples=1000, seed=config["seed"])
        precision, recall, f1, support = precision_recall_fscore_support(
            yTrue, yPred, labels=labels, zero_division=0
        )
        perClass = {
            lab: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for lab, p, r, f, s in zip(labels, precision, recall, f1, support)
        }
        cm = confusion_matrix(yTrue, yPred, labels=labels).tolist()

        results[rung] = {
            "macroF1": float(macroF1),
            "perClass": perClass,
            "confusion": {"labels": labels, "matrix": cm},
            "macroF1CI": [ciLow, ciHigh],
            "n": len(valPairs),
        }

    return results

def faithfulnessMetrics(config: dict) -> dict:
    raise NotImplementedError

def abstentionMetrics(config: dict) -> dict:
    raise NotImplementedError

def calibration(config: dict) -> dict:
    raise NotImplementedError

def latency(config: dict) -> dict:
    raise NotImplementedError

if __name__ == "__main__":
    cfg = loadConfig()
    setSeeds(cfg["seed"])
    runEval(cfg)

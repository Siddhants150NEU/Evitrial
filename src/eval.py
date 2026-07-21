from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import traceback
import bm25s, ir_measures
import logging
from ir_measures import MAP, R, calc_aggregate, nDCG
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

    measures = [nDCG@10, R@10, R@20, R@50, MAP]
    results = {}
    for name, run in [("bm25", bm25Run()), ("dense", denseRun())]:
        agg = calc_aggregate(measures, qrels, run)
        results[name] = {str(m): round(float(v), 4) for m, v in agg.items()}
    return results

def criterionMetrics(config: dict) -> dict:
    raise NotImplementedError

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

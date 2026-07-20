from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import traceback

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
    raise NotImplementedError

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

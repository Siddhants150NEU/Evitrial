"""The demo server. Stdlib only — no FastAPI, no Flask, nothing new in requirements.

Serves the front-end AND the JSON API from one origin, which sidesteps CORS and the
file:// restrictions that make a static deck annoying to develop against.

  python -m webapp.server.serveDemo            # port 8777, cached runs only
  python -m webapp.server.serveDemo --live     # also allow real runs (slow first boot)

ROUTES
  GET  /                    the app
  GET  /api/matchers        which rungs exist right now (see matcherRegistry)
  GET  /api/cache           list of pre-computed runs
  GET  /api/eval[?runId=]   per-rung numbers from a logged eval run
  GET  /api/cache/<id>      one pre-computed run
  POST /api/run             {note, rung, k} -> a real run. Only with --live.

WHY CACHED BY DEFAULT: a cold boot builds a BM25 index over 375,580 trials. That is
not something to do in front of an audience. Cached runs are instant and cannot fail;
--live is for the volunteer who wants to type their own note and can wait.
"""
from __future__ import annotations

import argparse
import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .paths import CACHE, EVAL_RUN_ID, RUNS, WEBAPP   # one definition, no heavy imports

_state = {"live": False, "config": None, "busy": threading.Lock()}
_summaries: dict[str, tuple] = {}           # path -> (mtime, size, summary)

def _summarise(f: Path) -> dict:
    """IN: a cache file. OUT: the small dict the picker needs.

    Memoised on (mtime, size) because the alternative is re-parsing every full run —
    about a megabyte today — each time somebody lands on the Start tab.
    """
    stamp = (f.stat().st_mtime_ns, f.stat().st_size)
    hit = _summaries.get(f.name)
    if hit and hit[:2] == stamp:
        return hit[2]

    d = json.loads(f.read_text())
    rows = [r for t in d.get("trials", []) for r in t.get("rows", [])]
    labels = Counter(r["verified"]["label"] for r in rows)
    # Contract failures are the generative rung's own abstention channel — llmContract
    # rejecting a malformed or off-target verdict, upstream of verify(). Counted here so
    # the Matchers tab can compare rungs without downloading every full run.
    contractFailures = sum(len(r["verified"].get("extra", {}).get("failures") or [])
                           for r in rows)

    # Sentences ride along so the picker can show the note without pulling the whole
    # run down first. Everything here is read by the front end — nothing speculative.
    summary = {"id": f.stem, "topicId": d.get("topicId"),
               "rung": d.get("rung"), "k": d.get("k"),
               "trials": len(d.get("trials", [])),
               "criteria": len(rows),
               "labels": dict(labels),
               "contractFailures": contractFailures,
               "forcedAbstentions": d.get("forcedAbstentions"),
               # matchMs, not totalMs, is the matcher's own cost. totalMs folds in
               # retrieval — including the ~135s cold BM25 build the first run in a
               # process pays — which would make whichever rung ran first look slowest.
               "matchMs": sum(t.get("matchMs") or 0 for t in d.get("trials", [])),
               "totalMs": d.get("totalMs"), "builtAt": d.get("builtAt"),
               "sentences": d.get("sentences", [])}
    _summaries[f.name] = (*stamp, summary)
    return summary

def _evalRun(runId: str) -> dict:
    """IN: a runId. OUT: the per-rung eval numbers the Matchers tab quotes.

    Reads reports/runs/<runId>/ — the only place a quotable number is allowed to come
    from. Deliberately drops the bulky bits (128-point risk-coverage curves, 10-bin
    calibration histograms): the tab shows headline figures, and shipping ~50 KB of
    curve data nobody plots would be silly.
    """
    if not runId.replace("-", "").replace("_", "").isalnum():      # no path games
        return {"error": "bad runId"}
    folder = RUNS / runId
    metricsFile, metaFile = folder / "metrics.json", folder / "meta.json"
    if not metricsFile.is_file():
        return {"error": f"no logged run {runId!r} under reports/runs/"}

    m = json.loads(metricsFile.read_text())
    meta = json.loads(metaFile.read_text()) if metaFile.is_file() else {}
    rungs = {}
    for rung, crit in (m.get("criterion") or {}).items():
        if not isinstance(crit, dict) or "macroF1" not in crit:
            rungs[rung] = {"status": crit.get("status") if isinstance(crit, dict) else "missing"}
            continue
        abst = (m.get("abstention") or {}).get(rung) or {}
        cal = (m.get("calibration") or {}).get(rung) or {}
        faith = (m.get("faithfulness") or {}).get(rung) or {}
        rungs[rung] = {
            "macroF1": crit["macroF1"], "macroF1CI": crit.get("macroF1CI"), "n": crit.get("n"),
            "perClass": {k: {"f1": v.get("f1"), "support": v.get("support")}
                         for k, v in (crit.get("perClass") or {}).items()},
            "coverage": abst.get("coverage"),
            "selectiveAccuracy": abst.get("selectiveAccuracy"),
            "unknownRecall": abst.get("unknownRecall"),
            "ece": cal.get("ece"), "brier": cal.get("brier"), "calN": cal.get("n"),
            "faithfulness": faith.get("faithfulness"),
            "forcedAbstentions": faith.get("forcedAbstentions"),
            "overreach": faith.get("overreach"),
            "failureCounts": faith.get("failureCounts") or {},
        }
    return {"runId": meta.get("runId", runId), "gitShortSha": meta.get("gitShortSha"),
            "seed": meta.get("seed"), "utc": meta.get("utc"),
            "rungs": rungs, "gateCalibration": m.get("gateCalibration")}

def _config() -> dict:
    """Load configs/default.yaml once and hang onto it. IN: nothing. OUT: the config dict."""
    if _state["config"] is None:
        from src.config import loadConfig, setSeeds
        cfg = loadConfig()
        setSeeds(cfg["seed"])
        _state["config"] = cfg
    return _state["config"]

class Handler(BaseHTTPRequestHandler):

    def do_GET(self):                                    # noqa: N802 (stdlib's name)
        path = self.path.split("?")[0]
        if path.startswith("/api/"):
            return self._api(path)
        return self._static(path)

    def do_POST(self):                                   # noqa: N802
        if self.path.split("?")[0] != "/api/run":
            return self._json({"error": "no such endpoint"}, 404)
        if not _state["live"]:
            return self._json({"error": "live runs are off. Restart with --live.",
                               "hint": "cached runs are at /api/cache"}, 503)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"] or 0)) or b"{}")
        note = (body.get("note") or "").strip()
        if not note:
            return self._json({"error": "give me a patient note"}, 400)

        # One run at a time. Two people hitting Run at once would thrash the models
        # and give both of them a bad time.
        if not _state["busy"].acquire(blocking=False):
            return self._json({"error": "already running a note, try again in a moment"}, 429)
        try:
            from .runStages import runNote
            result = runNote(note, _config(),
                             rung=body.get("rung", "zeroShot"),
                             k=int(body.get("k", 10)))
            result["source"] = "live"
            return self._json(result)
        except Exception as exc:                          # surface it, don't fake a result
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        finally:
            _state["busy"].release()

    # ---- routes ---------------------------------------------------------------
    def _api(self, path: str):
        if path == "/api/matchers":
            from .matcherRegistry import discoverRungs
            return self._json({"rungs": discoverRungs(), "live": _state["live"]})

        if path == "/api/eval":
            qs = parse_qs(urlparse(self.path).query)
            return self._json(_evalRun(qs.get("runId", [EVAL_RUN_ID])[0]))

        if path == "/api/cache":
            return self._json({"runs": [_summarise(f) for f in sorted(CACHE.glob("*.json"))]})

        if path.startswith("/api/cache/"):
            f = CACHE / f"{path.rsplit('/', 1)[-1]}.json"
            if not f.is_file() or f.parent != CACHE:      # no path traversal, thanks
                return self._json({"error": "no such cached run"}, 404)
            # The file already IS the response body. Parsing 350 KB of JSON only to
            # re-serialise it unchanged is pure ceremony.
            return self._send(f.read_bytes(), "application/json")

        return self._json({"error": "no such endpoint"}, 404)

    def _static(self, path: str):
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (WEBAPP / rel).resolve()
        if not str(target).startswith(str(WEBAPP)) or not target.is_file():
            return self._json({"error": "not found"}, 404)
        types = {".html": "text/html", ".css": "text/css", ".js": "text/javascript",
                 ".json": "application/json", ".svg": "image/svg+xml"}
        return self._send(target.read_bytes(),
                          types.get(target.suffix, "application/octet-stream"))

    def _json(self, obj: dict, status: int = 200):
        return self._send(json.dumps(obj).encode(), "application/json", status)

    def _send(self, body: bytes, contentType: str, status: int = 200):
        """The one place a response gets written, so headers can't drift per route."""
        self.send_response(status)
        self.send_header("Content-Type", contentType)
        # no-store everywhere: a run you just built should show up on reload, and an
        # edited .js/.css shouldn't need a hard refresh.
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):                    # noqa: N802 — quieter console
        print(f"  {self.command} {self.path}")

def main() -> None:
    ap = argparse.ArgumentParser(description="EVI-TRIAL demo server")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--live", action="store_true",
                    help="allow real pipeline runs (first one is slow: builds BM25)")
    args = ap.parse_args()
    _state["live"] = args.live

    print(f"\n  EVI-TRIAL demo  ->  http://localhost:{args.port}")
    print(f"  cached runs: {len(list(CACHE.glob('*.json')))}"
          f"   live runs: {'ON (first one is slow)' if args.live else 'off'}\n")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()

if __name__ == "__main__":
    main()

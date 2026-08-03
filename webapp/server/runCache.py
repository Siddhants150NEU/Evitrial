"""Pre-compute runs so the demo is instant and cannot fail on stage.

  python -m webapp.server.runCache --list
  python -m webapp.server.runCache --patient sigir-20141 --k 10
  python -m webapp.server.runCache --patient sigir-20141 --patient sigir-20142 --k 10

EACH PATIENT GETS ITS OWN CACHE FILE, always — one run, one file, independently
selectable in the UI. But pass them to ONE invocation rather than launching a process
per patient: the BM25 index (about 135s) and the matcher weights are built once and
reused across every patient in the same process. Separate processes re-pay both, every
time, for identical output.

NOTES COME FROM THE ANNOTATION SET, not the TREC topics. Two reasons: those notes are
already numbered ("0. ... 1. ..."), which is what splitNumberedNote expects; and they
carry expert labels, so the UI can show gold beside the prediction on the trials where
the two overlap. Gold is only ever DISPLAYED here — nothing writes a prediction back
into an annotation.

Writes webapp/cache/<topicId>__<rung>__k<k>.json, which is exactly what /api/cache serves
and exactly what a --live run returns. Same shape either way, so the front-end never has
to know which one it got.

BUDGET YOUR TIME. Measured on this machine, patient sigir-20141, rung zeroShot:

    k=3    633s   (136s retrieve + 497s match over 126 criteria)
    k=10  1107s   (136s retrieve + 975s match over 265 criteria)

Retrieval is a fixed cost paid once per process. Matching is ~3.7s per criterion and
scales with k, so budget roughly 15-20 minutes per extra patient at k=10.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src import ingest
from src.config import loadConfig, setSeeds

from .paths import CACHE
from .runStages import runNote

def loadPatients() -> dict[str, dict]:
    """IN: nothing. OUT: patientId -> {note, gold}, where gold is nctId -> criterionText -> label.

    Gold is read-only decoration for the UI. It never touches a Decision.
    """
    people: dict[str, dict] = {}
    for row in ingest.loadAnnotations(split="train"):
        p = people.setdefault(row["patientId"], {"note": row["note"], "gold": {}})
        p["gold"].setdefault(row["nctId"], {})[row["criterionText"]] = \
            ingest.mapToEviLabels(row["expertEligibility"])
    return people

def buildOne(topicId: str, note: str, config: dict, rung: str, k: int,
             gold: dict | None = None) -> Path:
    """IN: one patient + how to run it. OUT: the path we wrote. Prints progress as it goes."""
    print(f"  running {topicId}  rung={rung} k={k} ...", flush=True)
    started = time.time()
    result = runNote(note, config, rung=rung, k=k,
                     onStage=lambda e: print(f"    {e['id']:<9} {e['ms']:>6} ms", flush=True))
    result["topicId"] = topicId
    result["source"] = "cached"
    result["gold"] = gold or {}
    result["builtAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # k goes in the filename. Without it, rebuilding the same patient at a different k
    # silently overwrites the old one — which is exactly what happened the first time.
    CACHE.mkdir(exist_ok=True)
    out = CACHE / f"{topicId}__{rung}__k{k}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"  wrote {out.name}  ({_size(out)}, {time.time() - started:.0f}s, "
          f"{result['forcedAbstentions']} forced abstentions)\n")
    return out

def main() -> None:
    ap = argparse.ArgumentParser(description="pre-compute demo runs")
    ap.add_argument("--patient", action="append", help="patient id, repeatable")
    ap.add_argument("--list", action="store_true", help="show patient ids and exit")
    ap.add_argument("--rung", default="zeroShot")
    ap.add_argument("--k", type=int, default=10, help="trials per note. keep this small.")
    args = ap.parse_args()

    people = loadPatients()
    if args.list:
        for pid, p in people.items():
            first = next((s for s in p["note"].splitlines() if s.strip()), "")
            print(f"  {pid:<16} {len(p['gold'])} annotated trials · {first[:80]}")
        return

    wanted = args.patient or []
    if not wanted:
        ap.error("pass --patient <id>, or --list to see what's available")

    config = loadConfig()
    setSeeds(config["seed"])
    for pid in wanted:
        if pid not in people:
            print(f"  ! no patient {pid!r}, skipping")
            continue
        buildOne(pid, people[pid]["note"], config, args.rung, args.k,
                 gold=people[pid]["gold"])

def _size(p: Path) -> str:
    kb = p.stat().st_size / 1024
    return f"{kb:.0f} KB" if kb < 1024 else f"{kb / 1024:.1f} MB"

if __name__ == "__main__":
    main()

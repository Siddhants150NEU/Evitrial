"""Run the pipeline, but narrate it.

pipeline.runPatient() returns the answer. That's correct for the pipeline and useless
for a demo, where the whole point is the middle of the story: what each stage ate, what
it produced, and the exact moment verify() rewrites a label.

So this walks the SAME modules in the SAME order and emits an event per stage. It does
not reimplement any logic — every real decision is still made by src/.

THE COST, STATED PLAINLY: this is a second copy of pipeline.runPatient()'s stage order,
and nothing enforces that the two agree. Change the order in pipeline.py and this file
goes quietly stale. Nothing catches it. That was judged a better trade than editing the
spine to add a demo callback, but it is a real debt, not a solved problem — an AST test
comparing the ids emitted here against the span() names in pipeline.py would close it.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict

from src import ingest, parse, rank, retrieval
from src.checkIngest import EXPECTED
from src import verify as verifyModule

from .matcherRegistry import decisionToDict, runMatcher

def runNote(note: str, config: dict, rung: str, k: int, onStage=None) -> dict:
    """IN: a patient note, config, which matcher rung, how many trials to keep.
    OUT: one big dict describing the whole run, stage by stage.

    `onStage(event)` fires as each stage finishes, so a live caller can stream progress
    instead of staring at a spinner. Pass None if you just want the final dict.
    """
    started = time.time()
    stages: list[dict] = []

    def emit(event: dict) -> None:
        stages.append(event)
        if onStage:
            onStage(event)

    # ---- 01 ingest: prose -> addressable sentences ------------------------------
    t = time.time()
    sentences = ingest.splitNumberedNote(note)
    emit({
        "id": "ingest", "fn": "ingest.splitNumberedNote", "ms": _ms(t),
        "in": {"note": note},
        "out": {"sentences": sentences, "count": len(sentences)},
    })

    # ---- 02 retrieve: 375,580 trials -> k candidates -----------------------------
    t = time.time()
    candidates = retrieval.retrieve(note, config, k)
    trials = retrieval.fetchTrials([c.nctId for c in candidates], config)
    # Carry the trial's own metadata on the candidate. Retrieval returns ids and scores;
    # a human reading the list needs to see WHAT was retrieved to judge whether it's any
    # good. That judgement is the whole reason this stage gets shown.
    candidateRows = []
    for rank_, c in enumerate(candidates, start=1):
        tr = trials.get(c.nctId)
        candidateRows.append({
            **asdict(c),
            "rank": rank_,
            "title": tr.title if tr else None,
            "condition": tr.condition if tr else None,
            "summary": (tr.summary or "")[:400] if tr else None,
            "fetched": tr is not None,
        })
    emit({
        "id": "retrieve", "fn": "retrieval.retrieve + fetchTrials", "ms": _ms(t),
        "in": {"k": k, "topN": config["retrieval"]["topN"],
               "alpha": config["retrieval"]["alpha"],
               "useRerank": config["retrieval"]["useRerank"],
               # one source for the corpus size: the counts checkIngest verifies against
               "corpus": EXPECTED["trials"]},
        "out": {"candidates": candidateRows,
                "fetched": sum(1 for c in candidateRows if c["fetched"])},
    })

    # ---- 03-06, per candidate trial ----------------------------------------------
    perTrial, allParsed, allRaw, allVerified = [], [], [], []
    rankSec = 0.0
    for cand in candidates:
        trial = trials.get(cand.nctId)
        if trial is None:                       # retrieval found an id fetch couldn't
            continue                            # resolve; skip rather than invent one

        t = time.time()
        criteria = parse.parseCriteria(trial)
        allParsed.append({
            "nctId": trial.nctId, "title": trial.title,
            "criteria": [asdict(c) for c in criteria], "ms": _ms(t),
        })

        rows, checkedForTrial, calledVia = [], [], None
        # Accumulate in float seconds and round ONCE at the end. Rounding to whole
        # milliseconds per criterion floors verify()'s sub-millisecond cost to zero
        # every time, so the total came out as a fake 0 — a truncation artifact
        # dressed up as a measurement.
        matchSec = verifySec = 0.0
        for criterion in criteria:
            t0 = time.time()
            raw, calledVia = runMatcher(rung, note, criterion, config)
            t1 = time.time()
            checked = verifyModule.verify(raw, note, criterion.text)
            # Timed separately. verify() runs inside this loop, so folding it into
            # matchMs would over-report the matcher and make the gate look free.
            matchSec += t1 - t0
            verifySec += time.time() - t1
            rows.append({
                "criterion": asdict(criterion),
                "raw": decisionToDict(raw),
                "verified": decisionToDict(checked),
                # THE money field: did the gate actually rewrite this one?
                "forced": raw.label != checked.label,
            })
            checkedForTrial.append(checked)
            allRaw.append(raw)
            allVerified.append(checked)

        # rank.aggregate stays the only thing that scores, and it still gets real
        # Decision objects so its `assert all(d.verified)` means something.
        tRank = time.time()
        scored = rank.aggregate(cand.nctId, checkedForTrial, cand.score, config)
        rankSec += time.time() - tRank
        perTrial.append({
            "nctId": trial.nctId, "title": trial.title, "condition": trial.condition,
            "retrievalScore": cand.score, "retrieverBreakdown": cand.retrieverBreakdown,
            "rows": rows, "score": scored.score, "missingInfo": scored.missingInfo,
            "matchMs": round(matchSec * 1000), "verifyMs": round(verifySec * 1000),
            "calledVia": calledVia,
        })

    emit({
        "id": "parse", "fn": "parse.parseCriteria", "ms": sum(p["ms"] for p in allParsed),
        "in": {"trials": len(allParsed)},
        "out": {"perTrial": allParsed,
                "total": sum(len(p["criteria"]) for p in allParsed)},
    })

    forced = sum(1 for tr in perTrial for r in tr["rows"] if r["forced"])
    emit({
        "id": "match", "fn": f"match via {rung}", "ms": sum(p["matchMs"] for p in perTrial),
        "in": {"rung": rung, "calledVia": perTrial[0]["calledVia"] if perTrial else None},
        "out": {"decisions": len(allRaw),
                "labels": dict(Counter(d.label for d in allRaw))},
    })
    emit({
        "id": "verify", "fn": "verify.verify", "ms": sum(p["verifyMs"] for p in perTrial),
        "in": {"decisions": len(allRaw)},
        "out": {"forcedAbstentions": forced,
                "labels": dict(Counter(d.label for d in allVerified)),
                "supportedRate": round(1 - forced / len(allRaw), 4) if allRaw else None},
    })

    perTrial.sort(key=lambda p: p["score"], reverse=True)
    emit({
        "id": "rank", "fn": "rank.aggregate", "ms": round(rankSec * 1000),
        "in": {"weights": config["rank"]},
        "out": {"ranked": [{"nctId": p["nctId"], "score": p["score"],
                            "missing": len(p["missingInfo"])} for p in perTrial]},
    })

    return {
        "note": note,
        "sentences": sentences,
        "rung": rung,
        "k": k,
        "stages": stages,
        "trials": perTrial,
        "totalMs": _ms(started),
        "forcedAbstentions": forced,
    }

def _ms(since: float) -> int:
    return int((time.time() - since) * 1000)

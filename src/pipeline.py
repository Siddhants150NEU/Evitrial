from __future__ import annotations

from . import match, parse, rank, retrieval
from . import verify as verifyModule
from .schemas import TrialScore
from .trace import span

def runPatient(note: str, config: dict, k: int | None = None) -> list[TrialScore]:
    k = k or config["retrieval"]["k"]

    with span("retrieve"):
        candidates = retrieval.retrieve(note, config, k)

    trials = retrieval.fetchTrials([c.nctId for c in candidates], config)

    scored: list[TrialScore] = []
    for cand in candidates:
        trial = trials[cand.nctId]

        with span("parse"):
            criteria = parse.parseCriteria(trial)

        decisions = []
        for criterion in criteria:
            with span("match"):
                decision = match.match(note, criterion, config)
            with span("verify"):
                decision = verifyModule.verify(decision, note, criterion.text)
            decisions.append(decision)

        with span("aggregate"):
            scored.append(rank.aggregate(cand.nctId, decisions, cand.score, config))

    scored.sort(key=lambda ts: ts.score, reverse=True)
    return scored

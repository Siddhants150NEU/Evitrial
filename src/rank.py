from __future__ import annotations

from .schemas import Decision, TrialScore

def aggregate(nctId: str, decisions: list[Decision], retrievalScore: float, config: dict) -> TrialScore:
    assert all(d.verified for d in decisions), "unverified decision reached ranking (verify() was skipped)"

    w = config["rank"]
    incMet = sum(1 for d in decisions if d.criterionType == "inclusion" and d.label == "MET")
    incFail = sum(1 for d in decisions if d.criterionType == "inclusion" and d.label == "NOT_MET")
    excHit = sum(1 for d in decisions if d.criterionType == "exclusion" and d.label == "MET")

    score = (
        w["retrievalPrior"] * retrievalScore
        + w["inclusionMet"] * incMet
        - w["inclusionFailed"] * incFail
        - w["exclusionHit"] * excHit
    )

    missingInfo = [
        (d.criterionId or d.trialSpan) for d in decisions if d.label == "UNKNOWN"
    ]
    return TrialScore(nctId=nctId, score=score, decisions=decisions, missingInfo=missingInfo)

from __future__ import annotations

from dataclasses import replace

from .schemas import Decision, normalize

def isSupported(decision: Decision, note: str, criterionText: str) -> bool:
    if decision.label == "UNKNOWN":
        return True

    patientOk = bool(decision.patientSpan) and normalize(decision.patientSpan) in normalize(note)
    trialOk = bool(decision.trialSpan) and normalize(decision.trialSpan) in normalize(criterionText)
    return patientOk and trialOk

def verify(decision: Decision, note: str, criterionText: str) -> Decision:
    if decision.label != "UNKNOWN" and not isSupported(decision, note, criterionText):
        return replace(decision, label="UNKNOWN", patientSpan=None, verified=True)
    return replace(decision, verified=True)

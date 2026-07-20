from __future__ import annotations

import re
from dataclasses import dataclass, field

_WHITESPACE = re.compile(r"\s+")

def normalize(text: str | None) -> str:
    if not text:
        return ""
    return _WHITESPACE.sub(" ", text).strip().lower()

@dataclass
class Trial:

    nctId: str
    title: str
    condition: str
    summary: str
    detailedDescription: str
    eligibility: str
    inclusionCriteria: list[str] = field(default_factory=list)
    exclusionCriteria: list[str] = field(default_factory=list)

    def searchText(self) -> str:
        parts = [self.title, self.condition, self.summary, self.detailedDescription]
        return "\n".join(p for p in parts if p)

@dataclass
class Criterion:

    criterionId: str
    nctId: str
    text: str
    criterionType: str
    negation: bool = False
    temporal: str | None = None

@dataclass
class Candidate:

    nctId: str
    score: float
    retrieverBreakdown: dict = field(default_factory=dict)

@dataclass
class Decision:

    label: str
    confidence: float
    trialSpan: str
    patientSpan: str | None = None
    criterionId: str | None = None
    criterionType: str | None = None
    verified: bool = False

@dataclass
class TrialScore:

    nctId: str
    score: float
    decisions: list[Decision] = field(default_factory=list)
    missingInfo: list[str] = field(default_factory=list)

@dataclass
class PatientCriterionPair:

    patientId: str
    nctId: str
    criterionId: str
    note: str
    criterionText: str
    criterionType: str
    label: str
    patientSpan: str | None = None

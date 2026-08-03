from __future__ import annotations
from pydantic import BaseModel, ValidationError, Field, ConfigDict
import json
import logging
logger = logging.getLogger(__name__)

failureClasses = (
    "invalidJson", 
    "schemaViolation",
    "unknownLabel", 
    "confidenceOutOfRange",
    "indexOutOfRange", 
    "emptyIndices", 
    "wrongCriterion", 
    "emptyRationale", 
    "rationaleUngrounded",
    )

from typing import Literal, Annotated

class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criterionId: str
    label: Literal["MET", "NOT_MET", "UNKNOWN"]
    sentenceIndices: list[Annotated[int, Field(ge=0)]]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    
def parseVerdict(raw:str)-> tuple[Verdict|None, list[str]]:
    try:
        loaded = json.loads(raw)
    except (json.JSONDecodeError, TypeError) :
        return None, ["invalidJson"]
    
    if not isinstance(loaded, dict):
        return None, ["schemaViolation"]
    
    try:
        verdict = Verdict(**loaded)
        return verdict, []
    except ValidationError as e:
        fieldErrorMap = {
            "label": "unknownLabel",
            "confidence": "confidenceOutOfRange"
        }
        failure = set()
        for err in e.errors():
            loc = err.get("loc", [])
            locKey = str(loc[0]) if loc else ""
            failure.add(fieldErrorMap.get(locKey, "schemaViolation"))
        return None, sorted(list(failure))
    
def checkVerdict(verdict: 'Verdict', expectedCriterionId: str, sentences: list[str]) -> list[str]:
    failures = []

    if verdict.criterionId != expectedCriterionId:
        failures.append("wrongCriterion")

    if not verdict.rationale.strip():
        failures.append("emptyRationale")

    if verdict.label != "UNKNOWN" and not verdict.sentenceIndices:
        failures.append("emptyIndices")

    if verdict.sentenceIndices and any(i >= len(sentences) for i in verdict.sentenceIndices):
        failures.append("indexOutOfRange")

    return failures

def buildPrompt(criterionId: str, criterionText: str, sentences: list[str]) -> str:
    numbered_notes = "\n".join(f"  {i}: {sentence}" for i, sentence in enumerate(sentences))
    
    # prompt = f"""
    #     Patient note, one sentence per line: 
    #     {numbered_notes}
    #     Criterion (id: {criterionId}): {criterionText}

    #     Answer with JSON only:
    #     {{
    #     "criterionId": "...",
    #     "label": "MET"|"NOT_MET"|"UNKNOWN",
    #     "sentenceIndices": [<line numbers>],
    #     "confidence": 0.0-1.0,
    #     "rationale": "..."
    #     }}

    #     Rules:
    #     - Echo the exact criterionId back in the criterionId field.
    #     - Cite line NUMBERS in sentenceIndices. Never quote or paraphrase the note.
    #     - Use UNKNOWN when the note does not say. Not knowing is a correct answer — do not infer, do not assume typical patients.
    # """
    prompt = f"""Patient note, one sentence per line:
{numbered_notes}

Criterion (id: {criterionId}): {criterionText}

Answer with JSON only:
{{
"criterionId": "...",
"label": "MET"|"NOT_MET"|"UNKNOWN",
"sentenceIndices": [<line numbers>],
"confidence": 0.0-1.0,
"rationale": "..."
}}

Rules:
- Echo the exact criterionId back in the criterionId field.
- Cite line NUMBERS in sentenceIndices. Never quote or paraphrase the note.
- Use UNKNOWN when the note does not say. Not knowing is a correct answer — do not infer, do not assume typical patients.
"""

    return prompt


    
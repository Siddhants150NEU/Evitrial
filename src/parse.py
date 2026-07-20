from __future__ import annotations

from .schemas import Criterion, Trial

def parseCriteria(trial: Trial) -> list[Criterion]:
    raise NotImplementedError("implement atomic criterion splitting")

def detectCues(text: str) -> tuple[bool, str | None]:
    raise NotImplementedError("implement negation + temporal cue detection")

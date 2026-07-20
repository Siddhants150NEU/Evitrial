from __future__ import annotations

from .schemas import Criterion, Decision

def match(note: str, criterion: Criterion, config: dict) -> Decision:
    raise NotImplementedError("dispatch on config['matcher']['rung'] to the rungs below")

def ruleMatch(note: str, criterion: Criterion, config: dict) -> Decision:
    raise NotImplementedError("implement the lexical rule baseline")

def zeroShotMatch(note: str, criterion: Criterion, config: dict) -> Decision:
    raise NotImplementedError("implement the zero-shot NLI baseline")

def loraMatch(note: str, criterion: Criterion, config: dict) -> Decision:
    raise NotImplementedError("implement the LoRA-fine-tuned matcher")

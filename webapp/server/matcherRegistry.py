"""Find every matcher rung that exists in src/match.py, without being told about it.

WHY THIS EXISTS: a generative matcher is landing any day now. Nobody should have to
edit the UI to make it appear. Drop a `generativeMatch()` into match.py and this file
notices on the next server boot.

THE CONVENTION: a rung named `foo` is a module-level function called `fooMatch`.
That already describes ruleMatch / zeroShotMatch / loraMatch, so it costs nothing to
keep following it.

WHY AST AND NOT import: discovery only needs function NAMES. Importing src.match to get
them drags in torch, transformers and peft — about 4 seconds and three nltk downloads —
on every server that serves the picker, including the cached-only mode where no model is
ever touched. So we read match.py as text. Same trick testContracts.py uses to prove
retrieval never mentions qrels. The heavy import happens in runMatcher(), where a model
is genuinely about to run.
"""
from __future__ import annotations

import ast
import functools
from dataclasses import asdict
from pathlib import Path

from src.schemas import Criterion, Decision

MATCH_PY = Path(__file__).resolve().parents[2] / "src" / "match.py"

# Nicer names + one-liners for the rungs we already know about. Anything discovered
# that ISN'T in here still shows up — it just gets a title-cased label and no blurb.
# Adding a row here is the whole job of "make the new matcher look good in the UI".
KNOWN: dict[str, dict[str, str]] = {
    "rules": {
        "label": "Rules",
        "tag": "the floor",
        "blurb": "Lexical overlap, no model. The number everything else has to beat.",
    },
    "zeroShot": {
        "label": "Zero-shot",
        "tag": "off the shelf",
        "blurb": "A biomedical cross-encoder used as-is. No training on our data at all.",
    },
    "lora": {
        "label": "LoRA",
        "tag": "fine-tuned",
        "blurb": "DeBERTa-v3-large with a rank-16 adapter, trained on the train split only.",
    },
    "generative": {
        "label": "Generative",
        "tag": "incoming",
        "blurb": "An LLM that writes its reasoning. The first rung that will paraphrase — "
                 "which is exactly when the verify gate starts earning its keep.",
    },
}

# `rules` doesn't follow the convention (the function is ruleMatch, not rulesMatch),
# so it gets one hardcoded alias. Everything else derives cleanly.
_ALIAS = {"ruleMatch": "rules"}

_ORDER = ["rules", "zeroShot", "lora", "generative"]

@functools.lru_cache(maxsize=1)
def _readMatchModule() -> tuple[list[str], set[str], bool]:
    """IN: nothing. OUT: (matcher function names, rungs match() names explicitly, has catch-all else).

    Cached: match.py can't change under a running process, and re-parsing it per
    criterion would be silly.
    """
    tree = ast.parse(MATCH_PY.read_text())
    fnNames, named, catchAll = [], set(), False

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.endswith("Match") and not node.name.startswith("_"):
            fnNames.append(node.name)
        if node.name != "match":
            continue
        # Which rung strings does the dispatch actually test for, and does it end in a
        # bare `else` that swallows everything it didn't name?
        for sub in ast.walk(node):
            if isinstance(sub, ast.Compare):
                named.update(c.value for c in [sub.left, *sub.comparators]
                             if isinstance(c, ast.Constant) and isinstance(c.value, str))
            if isinstance(sub, ast.If) and sub.orelse and not isinstance(sub.orelse[0], ast.If):
                catchAll = True
    return fnNames, named, catchAll

def discoverRungs() -> list[dict]:
    """IN: nothing. OUT: one dict per matcher rung that actually exists right now.

    Each dict: rung, label, tag, blurb, fnName, dispatched (does match() name this rung
    explicitly?), catchAll (would match() silently route it somewhere else if not?).
    """
    fnNames, named, catchAll = _readMatchModule()
    found = []
    for name in fnNames:
        rung = _ALIAS.get(name) or name[: -len("Match")]
        meta = KNOWN.get(rung, {})
        found.append({
            "rung": rung,
            "label": meta.get("label", rung[:1].upper() + rung[1:]),
            "tag": meta.get("tag", "new"),
            "blurb": meta.get("blurb", "Discovered in match.py. Add a row to "
                                       "matcherRegistry.KNOWN to describe it."),
            "fnName": name,
            "dispatched": rung in named,
            "catchAll": catchAll,
        })
    found.sort(key=lambda r: (_ORDER.index(r["rung"]) if r["rung"] in _ORDER else 99, r["rung"]))
    return found

def runMatcher(rung: str, note: str, criterion: Criterion, config: dict) -> tuple[Decision, str]:
    """IN: a rung + the usual match() args. OUT: (raw Decision, how we called it).

    Goes through match.match() ONLY when match() names this rung explicitly. That caveat
    matters: match() currently ends in a bare `else` that forwards anything it doesn't
    recognise to loraMatch, so asking it for an unlisted rung returns LoRA output wearing
    the wrong name — silently, with no exception to catch. So for an unlisted rung we call
    its function directly and say so. The UI reports which path was taken, because
    pretending the two are the same would be a lie.
    """
    from src import match as matchModule        # heavy; only load it when we mean it

    cfg = {**config, "matcher": {**config["matcher"], "rung": rung}}
    entry = next((r for r in discoverRungs() if r["rung"] == rung), None)
    if entry is None:
        raise ValueError(f"no matcher rung called {rung!r} in match.py")

    if entry["dispatched"]:
        return matchModule.match(note, criterion, cfg), "match.match()"
    return getattr(matchModule, entry["fnName"])(note, criterion, cfg), f"{entry['fnName']}() directly"

# The Decision fields the UI lays out by name. Anything else the dataclass grows later —
# a `rationale` for the generative rung, say — lands in `extra` and renders generically.
_CORE = ("label", "confidence", "trialSpan", "patientSpan",
         "criterionId", "criterionType", "verified")

def decisionToDict(decision: Decision) -> dict:
    """IN: a Decision. OUT: a plain dict, including fields that didn't exist yesterday."""
    base = asdict(decision)
    return {**{k: base[k] for k in _CORE if k in base},
            "extra": {k: v for k, v in base.items() if k not in _CORE}}

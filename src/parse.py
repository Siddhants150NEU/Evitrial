from __future__ import annotations

import re
import string
from typing import Iterable

from .schemas import Criterion, Trial

_TERMINAL = (".", ";", ":", "!", "?")
_PUNCT = string.punctuation

_DANGLING = {
    "the", "a", "an", "of", "to", "for", "with", "and", "or", "in", "on",
    "at", "by", "from", "as", "including", "than", "into", "within", "without",
    "per", "between", "about", "over", "under", "during",
}

_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022\u25aa\u25e6]|\d+[.)]|[a-zA-Z][.)])\s+")


def _strip_bullet(text: str) -> str:
    return _BULLET_RE.sub("", text).strip()


_NEG_CUES = [
    r"no (?:history|evidence|signs?|symptoms?) of",
    r"negative for",
    r"absence of",
    r"free of",
    r"lack(?:ing|s)? of",
    r"ruled? out",
    r"unable to",
    r"fails? to",
    r"without",
    r"den(?:y|ies|ied)",
    r"non-\w+",
    r"cannot",
    r"never",
    r"none",
    r"not",
    r"no",
]
_NEG_RE = re.compile(r"\b(?:" + "|".join(_NEG_CUES) + r")\b", re.IGNORECASE)

_PSEUDO_NEG = re.compile(
    r"not only|no increase|no change|no further|not necessarily|"
    r"gram[- ]negative|not certain",
    re.IGNORECASE,
)


def detect_negation(text: str) -> tuple[bool, list[str]]:
    scrubbed = _PSEUDO_NEG.sub("  ", text)
    cues = [m.group(0).lower() for m in _NEG_RE.finditer(scrubbed)]
    return (len(cues) > 0, cues)


_NUM_WORD = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|a|an"
)
_QTY = rf"(?:\d+(?:\.\d+)?|{_NUM_WORD})"
_UNIT = (
    r"sec(?:ond)?s?|min(?:ute)?s?|hours?|hrs?|days?|weeks?|wks?|"
    r"months?|mos?|years?|yrs?"
)
_REL = (
    r"within the last|within the past|within|in the last|in the past|in th past|"
    r"over the last|over the past|for the last|for the past|prior to|at least|"
    r"at most|up to|no more than|no less than|more than|less than|greater than|"
    r"fewer than|before|after|during|since|every|following|preceding|until|for|"
    r">=|<=|>|<|\u2265|\u2264"
)

_DURATION_RE = re.compile(
    rf"(?:(?:{_REL})\s+)*{_QTY}\s+(?:{_UNIT})\b(?:\s+ago)?",
    re.IGNORECASE,
)

_RELATIVE_RE = re.compile(
    r"\b(?:in th(?:e)? past|previously|recent(?:ly)?|currently|ongoing|"
    r"history of)\b",
    re.IGNORECASE,
)


def detect_temporal(text: str) -> list[str]:
    spans: list[str] = []
    seen: set[str] = set()
    for rx in (_DURATION_RE, _RELATIVE_RE):
        for m in rx.finditer(text):
            s = m.group(0).strip()
            key = s.lower()
            if s and key not in seen:
                seen.add(key)
                spans.append(s)
    return spans


def detectCues(text: str) -> tuple[bool, str]:
    negated, _cues = detect_negation(text)
    return negated, "; ".join(detect_temporal(text))


_PROTECT_RE = re.compile(
    r"(?:great(?:er)?|less(?:er)?|more|fewer|higher|lower|older|younger|"
    r"earlier|later|bigger|smaller)\s+(?:than\s+)?or\s+equal(?:\s+to)?"
    r"|equal(?:\s+to)?\s+or\s+(?:great(?:er)?|less(?:er)?|more|fewer|higher|lower)"
    r"|\bor\s+equal\b|\bequal\s+or\b|\band\s*/\s*or\b"
    r"|\bsigns?\s+(?:and|or)\s+symptoms?\b"
    r"|\bnausea\s+and\s+vomiting\b"
    r"|\b\d+(?:\.\d+)?\s+(?:and|or|to)\s+\d+(?:\.\d+)?\b"
    r"|\bbetween\s+\d+(?:\.\d+)?\s+and\s+\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)

_SPLIT_CONJ_RE = re.compile(r"\s*,?\s+\b(?:and|or)\b\s+", re.IGNORECASE)

_DET = {"a", "an", "the", "any", "all", "each", "some", "this", "that",
        "these", "those", "no", "every", "either", "neither", "both", "one"}
_PREP = {"of", "in", "on", "during", "with", "for", "to", "from", "at", "by",
         "within", "without", "into", "over", "under", "per", "between",
         "about", "as", "than", "after", "before"}
_INDEP_START = _DET | {"not", "never", "none", "non"}
_ADJ_SUFFIX = re.compile(
    r"(?:ant|ent|ical|ic|ive|ous|ile|ory|ary|al|ing|ed|lar|ac|oid|otic|"
    r"emic|uric|genic)$",
    re.IGNORECASE,
)


def _first_word(s: str) -> str:
    m = re.search(r"[A-Za-z]+", s)
    return m.group(0).lower() if m else ""


def _prefix_boundary(tokens: list[str]) -> int:
    idx = -1
    for j, t in enumerate(tokens):
        w = t.strip(_PUNCT).lower()
        if w in _DET or w in _PREP:
            idx = j
    return idx


def _split_conjuncts(text: str) -> list[str]:
    masked_spans: list[str] = []

    def _mask(m: re.Match) -> str:
        masked_spans.append(m.group(0))
        return f"\x00{len(masked_spans) - 1}\x00"

    masked = _PROTECT_RE.sub(_mask, text)
    parts = [p.strip() for p in _SPLIT_CONJ_RE.split(masked) if p.strip()]
    if len(parts) < 2:
        return [text]

    def _unmask(s: str) -> str:
        for i, orig in enumerate(masked_spans):
            s = s.replace(f"\x00{i}\x00", orig)
        return s.strip()

    later = parts[1:]

    if any(_first_word(p) in _INDEP_START for p in later):
        return [_unmask(p) for p in parts]

    p0_tokens = parts[0].split()
    if len(p0_tokens) >= 3 and all(len(p.split()) <= 3 for p in later):
        bidx = _prefix_boundary(p0_tokens)
        if 0 <= bidx < len(p0_tokens) - 1:
            prefix = " ".join(p0_tokens[:bidx + 1])
            out = [parts[0]] + [f"{prefix} {p}" for p in later]
            return [_unmask(p) for p in out]

    last_tokens = parts[-1].split()
    if (len(last_tokens) >= 2
            and all(len(p.split()) == 1 and _ADJ_SUFFIX.search(p)
                    for p in parts[:-1])):
        head = last_tokens[-1]
        out = [f"{p} {head}" for p in parts[:-1]] + [parts[-1]]
        return [_unmask(p) for p in out]

    return [_unmask(p) for p in parts]


def _as_lines(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return raw.splitlines()
    return [str(x) for x in raw]


def _is_continuation(prev: str, cur: str, wrap_width: int) -> bool:
    p = prev.rstrip()
    if p.endswith(_TERMINAL):
        return False
    if cur[:1].islower():
        return True
    last = p.split()[-1].strip(_PUNCT).lower()
    if last in _DANGLING:
        return True
    return len(p) >= wrap_width


def _reflow(lines: Iterable[str], wrap_width: int) -> list[str]:
    out: list[str] = []
    buf = ""
    last = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if buf and _is_continuation(last, line, wrap_width):
            buf = f"{buf} {line}"
        else:
            if buf:
                out.append(buf)
            buf = line
        last = line
    if buf:
        out.append(buf)
    return out


_ABBREV = {"e.g", "i.e", "vs", "etc", "approx", "no"}
_SENT_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    merged: list[str] = []
    for part in parts:
        if merged:
            prev_last = merged[-1].split()[-1].rstrip(".").lower()
            if prev_last in _ABBREV:
                merged[-1] = f"{merged[-1]} {part}"
                continue
        merged.append(part)
    return [p.strip() for p in merged if p.strip()]


def parseCriteria(trial: Trial, *, wrap_width: int = 80,
                  split_sentences: bool = True,
                  split_conjuncts: bool = True) -> list[Criterion]:
    out: list[Criterion] = []
    for kind in ("inclusion", "exclusion"):
        raw = (trial.inclusionCriteria if kind == "inclusion"
               else trial.exclusionCriteria)

        logical = _reflow(_as_lines(raw), wrap_width)

        units: list[str] = []
        for crit in logical:
            crit = _strip_bullet(crit)
            sents = _split_sentences(crit) if split_sentences else [crit]
            for s in sents:
                units.extend(_split_conjuncts(s) if split_conjuncts else [s])

        i = 0
        for text in units:
            text = text.strip()
            if not text:
                continue
            negated, temporal = detectCues(text)
            out.append(Criterion(
                criterionId=f"{trial.nctId}::{kind[:3]}::{i}",
                nctId=trial.nctId,
                text=text,
                criterionType=kind,
                negation=negated,
                temporal=temporal,
            ))
            i += 1
    return out
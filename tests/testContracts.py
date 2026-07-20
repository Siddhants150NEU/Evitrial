from __future__ import annotations

import ast
import dataclasses
import pathlib

from src import rank, verify
from src.schemas import Decision, Trial, normalize

TINY_CONFIG = {
    "rank": {
        "retrievalPrior": 1.0,
        "inclusionMet": 1.0,
        "inclusionFailed": 1.0,
        "exclusionHit": 5.0,
    }
}

def testSchemaRoundTrip():
    trial = Trial(
        nctId="NCT00000001", title="t", condition="c", summary="s",
        detailedDescription="d", eligibility="e",
        inclusionCriteria=["a"], exclusionCriteria=["b"],
    )
    restored = Trial(**dataclasses.asdict(trial))
    assert restored == trial

def testNormalizeSubstringContract():
    note = "The  Patient  is 58 years old and denies diabetes."
    span = "58 YEARS old"
    assert normalize(span) in normalize(note)
    assert normalize("patient is") in normalize(note)

def testVerifyForcesUnknownOnMissingSpan():
    bad = Decision(
        label="MET", confidence=0.9,
        trialSpan="age >= 18", patientSpan="patient is 40 years old",
        criterionType="inclusion", verified=False,
    )
    out = verify.verify(bad, note="A 2-year-old boy with fever.", criterionText="age >= 18")
    assert out.label == "UNKNOWN"
    assert out.patientSpan is None
    assert out.verified is True

def testVerifyKeepsValidDecision():
    good = Decision(
        label="MET", confidence=0.8,
        trialSpan="fever", patientSpan="fever up to 39 C",
        criterionType="inclusion", verified=False,
    )
    out = verify.verify(good, note="An 8-year-old with fever up to 39 C.", criterionText="fever present at screening")
    assert out.label == "MET"
    assert out.verified is True
    assert out.patientSpan == "fever up to 39 C"

def testRankRejectsUnverified():
    unverified = Decision(label="MET", confidence=1.0, trialSpan="x", criterionType="inclusion", verified=False)
    raised = False
    try:
        rank.aggregate("NCT1", [unverified], retrievalScore=1.0, config=TINY_CONFIG)
    except AssertionError:
        raised = True
    assert raised, "aggregate() must reject decisions that were not verified"

def testRankAcceptsVerified():
    verified = Decision(label="MET", confidence=1.0, trialSpan="x", criterionType="inclusion", verified=True)
    score = rank.aggregate("NCT1", [verified], retrievalScore=2.0, config=TINY_CONFIG)
    assert score.nctId == "NCT1"
    assert score.score == 2.0 * 1.0 + 1.0

def testNoQrelLeakInRetrieval():
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "retrieval.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            names.add(node.attr.lower())
        elif isinstance(node, ast.alias):
            names.add(node.name.lower())
            if node.asname:
                names.add(node.asname.lower())
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.lower())
    offenders = sorted(n for n in names if "qrel" in n)
    assert offenders == [], f"retrieval.py must not reference qrels in code; found: {offenders}"

def _runAll():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
        except Exception as exc:
            failures += 1
            print(f"  [FAIL] {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} contract tests passed.")
    return failures

if __name__ == "__main__":
    import sys

    sys.exit(1 if _runAll() else 0)

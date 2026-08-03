"""Guardrails for the LLM airlock — src/llmContract.py.

WHY THIS FILE EXISTS: llmContract is the only module in the repo whose job is to
catch something else misbehaving. If it silently breaks, the ledger reports clean
runs forever and nobody finds out. An untested validator is worse than none,
because you trust it. So every failure class gets a test that proves it fires.

HOW TO READ IT: tests are grouped by gate.
  gate 1+2  parseVerdict  -- is it JSON, and is it the right shape?
  gate 3+5  checkVerdict  -- does it point at real sentences, for OUR question?
  vocabulary              -- does failureClasses match what the code actually says?

Run: pytest tests/testLlmContract.py   (or plain: python tests/testLlmContract.py)
"""
from __future__ import annotations

import json

from src.llmContract import Verdict, checkVerdict, failureClasses, parseVerdict

# A payload that should sail through untouched. Every bad case below is this,
# with exactly one thing broken -- so a failure always points at one cause.
GOOD = {
    "criterionId": "c1",
    "label": "MET",
    "sentenceIndices": [0],
    "confidence": 0.9,
    "rationale": "the note says the patient is 58",
}

SENTENCES = ["patient is 58 years old.", "denies diabetes.", "no prior chemotherapy."]

def _raw(**overrides) -> str:
    """IN: fields to break. OUT: a JSON string of GOOD with those fields replaced."""
    return json.dumps({**GOOD, **overrides})

def _verdict(**overrides) -> Verdict:
    """IN: fields to override. OUT: a valid Verdict. For testing gate 3+5 directly."""
    return Verdict(**{**GOOD, **overrides})

# ---------------------------------------------------------------- gate 1: transport
# "Is that an ID, or a napkin?" Nothing here is about MEANING yet -- only whether
# bytes off the wire are even parseable into a dict.

def testParseAcceptsCleanPayload():
    """The happy path. If this breaks, every other test is meaningless."""
    verdict, failures = parseVerdict(_raw())
    assert failures == [], failures
    assert verdict is not None
    assert verdict.label == "MET"

def testParseRejectsBrokenJson():
    """A truncated response -- the classic 'model hit the token cap' failure."""
    _, failures = parseVerdict("{")
    assert failures == ["invalidJson"], failures

def testParseRejectsNonString():
    """Guards the TypeError path: something upstream handed us None, not text."""
    _, failures = parseVerdict(None)
    assert failures == ["invalidJson"], failures

def testParseRejectsBareArray():
    """Valid JSON, wrong container. Without the isinstance guard this is an
    uncaught TypeError from Verdict(**list), not a failure class."""
    _, failures = parseVerdict("[1, 2, 3]")
    assert failures == ["schemaViolation"], failures

def testParseRejectsJsonNull():
    """json.loads("null") returns None, which is valid JSON and not a dict."""
    _, failures = parseVerdict("null")
    assert failures == ["schemaViolation"], failures

def testParseReturnsNoVerdictOnFailure():
    """A partial verdict must never escape. A stub that fakes a result is worse
    than one that refuses -- same rule as NotImplementedError in the stubs."""
    verdict, failures = parseVerdict("{")
    assert verdict is None
    assert failures

# ---------------------------------------------------------------- gate 2: shape
# "This says you were born in the year 3000." The JSON parsed; now does it obey
# the contract? Each of these is a lie a real model has told at some point.

def testParseRejectsUnknownLabel():
    """The one that matters most: an invented label like MAYBE or PROBABLY_MET.
    Without Literal[...] this walks straight through as a valid string."""
    _, failures = parseVerdict(_raw(label="MAYBE"))
    assert failures == ["unknownLabel"], failures

def testParseRejectsLowercaseLabel():
    """'met' is not MET. Literal is case-sensitive, and that's deliberate --
    silently upcasing would hide a model that isn't following the format."""
    _, failures = parseVerdict(_raw(label="met"))
    assert failures == ["unknownLabel"], failures

def testParseRejectsConfidenceAboveOne():
    """An out-of-range confidence silently poisons ECE and the reliability bins
    downstream -- the calibration plot looks fine and is wrong."""
    _, failures = parseVerdict(_raw(confidence=1.7))
    assert failures == ["confidenceOutOfRange"], failures

def testParseRejectsNegativeConfidence():
    assert parseVerdict(_raw(confidence=-0.2))[1] == ["confidenceOutOfRange"]

def testParseRejectsNegativeIndex():
    """THE SNEAKY ONE. sentences[-1] is legal Python and returns the last
    sentence, so a negative index would dereference successfully and ship a
    decision pointing at the wrong evidence, with no error anywhere."""
    _, failures = parseVerdict(_raw(sentenceIndices=[-1]))
    assert failures == ["schemaViolation"], failures

def testParseRejectsNonIntegerIndex():
    """A string index reaches checkVerdict and crashes it with a TypeError.
    Gate 3 shouldn't have to defend against garbage gate 2 exists to stop."""
    _, failures = parseVerdict(_raw(sentenceIndices=["banana"]))
    assert failures == ["schemaViolation"], failures

def testParseRejectsExtraField():
    """The model improvising a field it wasn't asked for. Silently ignoring it
    means never noticing the format drifted."""
    _, failures = parseVerdict(_raw(certainty=0.5))
    assert failures == ["schemaViolation"], failures

def testParseRejectsMissingField():
    payload = {k: v for k, v in GOOD.items() if k != "rationale"}
    _, failures = parseVerdict(json.dumps(payload))
    assert failures == ["schemaViolation"], failures

def testNegativeIndexIsNotIndexOutOfRange():
    """Gate 2 and gate 3 must stay distinguishable in the ledger. A malformed
    index is the model emitting garbage; an index of 99 on a 6-sentence note is
    the model pointing at nothing. Different diseases, different fixes."""
    _, failures = parseVerdict(_raw(sentenceIndices=[-1]))
    assert "indexOutOfRange" not in failures, failures

# ---------------------------------------------------------------- multi-failure + determinism

def testParseCollectsMultipleFailures():
    """Two things wrong must report as two. Reporting one makes the taxonomy
    undercount, and the numbers would look plausible while being wrong."""
    _, failures = parseVerdict(_raw(label="MAYBE", confidence=1.7))
    assert set(failures) == {"unknownLabel", "confidenceOutOfRange"}, failures

def testParseDeduplicatesFailures():
    """Two bad indices are one schemaViolation, not two."""
    _, failures = parseVerdict(_raw(sentenceIndices=[-1, -2]))
    assert failures == ["schemaViolation"], failures

def testParseFailureOrderIsDeterministic():
    """list(set(...)) reorders between runs. In a project whose whole claim is
    reproducible runs, a ledger that shuffles is a real defect."""
    payload = _raw(label="MAYBE", confidence=1.7, sentenceIndices=[-1])
    runs = [parseVerdict(payload)[1] for _ in range(5)]
    assert all(r == runs[0] for r in runs), runs
    assert runs[0] == sorted(runs[0]), runs[0]

# ---------------------------------------------------------------- gate 3: grounding
# "Seat 47? This theater has 12 seats."

def testCheckAcceptsCleanVerdict():
    assert checkVerdict(_verdict(), "c1", SENTENCES) == []

def testCheckRejectsIndexPastEnd():
    assert checkVerdict(_verdict(sentenceIndices=[99]), "c1", SENTENCES) == ["indexOutOfRange"]

def testCheckRejectsIndexEqualToLength():
    """Boundary. len(SENTENCES) == 3, so index 3 is one past the last valid slot.
    Off-by-one here means silently dereferencing nothing, or an IndexError later."""
    assert checkVerdict(_verdict(sentenceIndices=[3]), "c1", SENTENCES) == ["indexOutOfRange"]

def testCheckAcceptsLastValidIndex():
    """The other side of the same boundary -- index 2 of 3 must pass."""
    assert checkVerdict(_verdict(sentenceIndices=[2]), "c1", SENTENCES) == []

def testCheckRejectsAnyBadIndexInAList():
    """One rotten index spoils the verdict: the engine can't ship a decision
    whose evidence is partly imaginary."""
    assert checkVerdict(_verdict(sentenceIndices=[0, 99]), "c1", SENTENCES) == ["indexOutOfRange"]

def testCheckHandlesEmptySentenceList():
    """An empty note means every index is out of range. Must not crash."""
    assert checkVerdict(_verdict(), "c1", []) == ["indexOutOfRange"]

# ---------------------------------------------------------------- gate 5: cross-checks
# "This ticket is for a different show."

def testCheckRejectsWrongCriterion():
    """The scariest failure mode there is: a fluent, well-formatted, confident
    answer to a question you never asked. Costs one != to catch."""
    assert checkVerdict(_verdict(criterionId="c9"), "c1", SENTENCES) == ["wrongCriterion"]

def testCheckRejectsBlankRationale():
    """Whitespace is not a rationale. .strip() or this passes."""
    assert checkVerdict(_verdict(rationale="   "), "c1", SENTENCES) == ["emptyRationale"]

def testCheckRejectsConfidentLabelWithNoEvidence():
    """MET with zero cited sentences is a claim with nothing behind it."""
    assert checkVerdict(_verdict(sentenceIndices=[]), "c1", SENTENCES) == ["emptyIndices"]

def testCheckAllowsUnknownWithNoEvidence():
    """The mirror image, and NOT a failure. Abstaining without citing anything
    is exactly what we want a model to do when the note is silent."""
    assert checkVerdict(_verdict(label="UNKNOWN", sentenceIndices=[]), "c1", SENTENCES) == []

def testCheckAllowsUnknownWithEvidence():
    """Citing why you can't tell is also legitimate -- don't punish it."""
    assert checkVerdict(_verdict(label="UNKNOWN", sentenceIndices=[1]), "c1", SENTENCES) == []

def testCheckCollectsEveryFailure():
    """Three problems must report as three. If checkVerdict early-returns, the
    ledger systematically undercounts and nothing looks obviously wrong."""
    bad = _verdict(criterionId="c9", rationale=" ", sentenceIndices=[99])
    assert checkVerdict(bad, "c1", SENTENCES) == [
        "wrongCriterion", "emptyRationale", "indexOutOfRange",
    ]

def testCheckOrderIsStable():
    """Appended in fixed order, so no sorting needed -- but prove it stays that
    way, because a reordered ledger breaks run-to-run diffing."""
    bad = _verdict(criterionId="c9", rationale=" ", sentenceIndices=[99])
    runs = [checkVerdict(bad, "c1", SENTENCES) for _ in range(5)]
    assert all(r == runs[0] for r in runs), runs

# ---------------------------------------------------------------- the vocabulary itself

def testEveryEmittedClassIsInTheVocabulary():
    """THE DRIFT CATCHER. failureClasses exists so the ledger and the validator
    speak one language. This test is what makes that true instead of aspirational
    -- it caught a `schemaVoilation` typo that three human read-throughs missed.
    """
    emitted = set()
    for payload in ["{", "null", "[1,2]", _raw(label="MAYBE"),
                    _raw(confidence=1.7), _raw(sentenceIndices=[-1])]:
        emitted.update(parseVerdict(payload)[1])
    for verdict, expected in [(_verdict(criterionId="c9"), "c1"),
                              (_verdict(rationale=" "), "c1"),
                              (_verdict(sentenceIndices=[]), "c1"),
                              (_verdict(sentenceIndices=[99]), "c1")]:
        emitted.update(checkVerdict(verdict, expected, SENTENCES))

    unknown = sorted(emitted - set(failureClasses))
    assert unknown == [], f"emitted but missing from failureClasses: {unknown}"

def testVocabularyHasNoDuplicates():
    """A duplicated name double-counts in the ledger."""
    assert len(failureClasses) == len(set(failureClasses))

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
    print(f"\n{len(tests) - failures}/{len(tests)} llmContract tests passed.")
    return failures

if __name__ == "__main__":
    import sys

    sys.exit(1 if _runAll() else 0)

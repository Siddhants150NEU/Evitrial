from __future__ import annotations

import argparse
import sys
from collections import Counter

from . import ingest

EXPECTED = {
    "trials": 375_580,
    "topics": 75,
    "qrels": 35_832,
    "qrelRelevance": {0: 24_243, 1: 6_019, 2: 5_570},
    "annotationRows": 1_015,
    "annotationPatients": 53,
    "annotationTrials": 103,
    "criterionTypes": {"inclusion", "exclusion"},
    "eviLabels": {"MET", "NOT_MET", "UNKNOWN"},
}

_PASS = "  [PASS]"
_FAIL = "  [FAIL]"

def _line(ok: bool, label: str, got, expected=None) -> bool:
    tag = _PASS if ok else _FAIL
    detail = f"got {got}" + (f", expected {expected}" if expected is not None else "")
    print(f"{tag}  {label:<34} {detail}")
    return ok

def checkTrials(fast: bool) -> bool:
    print("\nTrials (TREC CT 2021 corpus)")
    dataset = ingest._dataset(ingest.TREC_2021)

    if fast:
        first = next(dataset.docs_iter(), None)
        ok = first is not None and str(first.doc_id).startswith("NCT")
        ok = _line(ok, "corpus loads, ids look like NCT*", getattr(first, "doc_id", None))
        print("         (skipped full count; run without --fast to verify 375,580)")
        return ok

    try:
        count = dataset.docs_count()
    except Exception:
        count = sum(1 for _ in dataset.docs_iter())
    return _line(count == EXPECTED["trials"], "corpus trial count", count, EXPECTED["trials"])

def checkTopics() -> bool:
    print("\nTopics (synthetic patient notes)")
    topics = list(ingest.loadTopics(ingest.TREC_2021))
    okCount = _line(len(topics) == EXPECTED["topics"], "topic count", len(topics), EXPECTED["topics"])
    nonEmpty = all(text.strip() for _, text in topics)
    okText = _line(nonEmpty, "every topic has note text", nonEmpty)
    return okCount and okText

def checkQrels() -> bool:
    print("\nQrels (relevance judgements)")
    dist = Counter(rel for _, _, rel in ingest.loadQrels(ingest.TREC_2021))
    total = sum(dist.values())
    okTotal = _line(total == EXPECTED["qrels"], "total judgements", total, EXPECTED["qrels"])
    okDist = _line(
        dict(dist) == EXPECTED["qrelRelevance"],
        "relevance distribution (0/1/2)",
        dict(sorted(dist.items())),
        EXPECTED["qrelRelevance"],
    )
    return okTotal and okDist

def checkAnnotations() -> bool:
    print("\nAnnotations (TrialGPT criterion labels -> matching eval)")
    rows = ingest.loadAnnotations()
    okRows = _line(len(rows) == EXPECTED["annotationRows"], "row count", len(rows), EXPECTED["annotationRows"])

    patients = {r["patientId"] for r in rows}
    trials = {r["nctId"] for r in rows}
    okPatients = _line(len(patients) == EXPECTED["annotationPatients"], "unique patients", len(patients), EXPECTED["annotationPatients"])
    okTrials = _line(len(trials) == EXPECTED["annotationTrials"], "unique trials", len(trials), EXPECTED["annotationTrials"])

    types = {r["criterionType"] for r in rows}
    okTypes = _line(types == EXPECTED["criterionTypes"], "criterion types", sorted(types), sorted(EXPECTED["criterionTypes"]))

    pairs = ingest.toEvalPairs(rows)
    labels = {p.label for p in pairs}
    okLabels = _line(labels <= EXPECTED["eviLabels"], "mapped labels within 3-class set", sorted(labels))

    grounded = sum(1 for p in pairs if p.label in {"MET", "NOT_MET"} and p.patientSpan)
    okSpans = _line(grounded > 0, "gold spans extracted for grounded pairs", f"{grounded} pairs")

    trainCount = sum(1 for r in rows if r["training"])
    print(f"         (info) training flag split: {trainCount} train / {len(rows) - trainCount} eval")
    labelBreakdown = Counter(p.label for p in pairs)
    print(f"         (info) label breakdown: {dict(labelBreakdown)}")
    return all([okRows, okPatients, okTrials, okTypes, okLabels, okSpans])

def main() -> int:
    parser = argparse.ArgumentParser(description="Verify EVI-TRIAL ingestion.")
    parser.add_argument("--fast", action="store_true", help="skip the slow full-corpus trial count")
    args = parser.parse_args()

    print("=" * 68)
    print("EVI-TRIAL ingestion check")
    print("=" * 68)

    results = [
        checkTrials(args.fast),
        checkTopics(),
        checkQrels(),
        checkAnnotations(),
    ]

    print("\n" + "=" * 68)
    if all(results):
        print("ALL CHECKS PASSED -- ingestion looks healthy. Safe to build downstream.")
        print("=" * 68)
        return 0
    print("SOME CHECKS FAILED -- STOP and fix ingestion before going further (see 05).")
    print("=" * 68)
    return 1

if __name__ == "__main__":
    sys.exit(main())

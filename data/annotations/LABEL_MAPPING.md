# Label mapping: TrialGPT (6 labels) → EVI-TRIAL (3 labels)

The matching data (`ncbi/TrialGPT-Criterion-Annotations`) labels each patient–criterion
pair with **6** possible expert labels. EVI-TRIAL uses **3**. This file is the single
explanation of how one becomes the other, so `ingest.mapToEviLabels` never becomes a
mystery.

## The core convention

**MET = "the criterion's *condition* is true of the patient."**

That's it. We do NOT bake "is the patient eligible" into MET/NOT_MET — eligibility is a
*ranking* decision that `rank.py` makes afterward (an inclusion the patient meets is
good; an exclusion whose condition is true is bad). Keeping MET purely about "does the
condition hold" makes the matcher a clean entailment task: does the note *entail* the
criterion condition?

| Criterion type | TrialGPT label            | Condition holds? | EVI label   |
| -------------- | ------------------------- | ---------------- | ----------- |
| inclusion      | `included`                | yes              | **MET**     |
| inclusion      | `not included`            | no (contradicted)| **NOT_MET** |
| exclusion      | `excluded`                | yes              | **MET**     |
| exclusion      | `not excluded`            | no (contradicted)| **NOT_MET** |
| either         | `not enough information`  | can't tell       | **UNKNOWN** |
| either         | `not applicable`          | doesn't apply    | **UNKNOWN** |

### Why `excluded` → MET (this trips people up)

`excluded` means the exclusion *condition* is true for the patient (e.g. criterion
"pregnant", patient is pregnant). The condition holds, so it's **MET**. Whether that's
*good* for the patient is a separate question — and it's bad — but that judgement is
`rank.py`'s job (`exclusionHit` penalty), not the matcher's. The matcher only reports
whether the condition is true.

### Why `not applicable` → UNKNOWN

"Not applicable" (e.g. a pregnancy criterion for a 2-year-old boy) means the condition
can't meaningfully be said to hold or not. We fold it into UNKNOWN so the matcher never
pretends to a MET/NOT_MET it can't justify. If you later want to treat N/A specially,
handle it as its own bucket — but do it explicitly, don't silently retag it.

## Gold evidence spans (free faithfulness labels)

Every row also has `expert_sentences`, the sentence indices the expert cited as
evidence. Patient notes are pre-numbered (`0. ... 1. ... 2. ...`), so
`expert_sentences = [3, 4]` means the gold evidence is sentences 3 and 4, joined.
`ingest.toEvalPairs` turns this into `PatientCriterionPair.patientSpan`, which is the
gold the verifier's supported-rate and the abstention metrics are scored against.

- MET / NOT_MET rows usually have a non-empty `expert_sentences` → a gold span exists.
- UNKNOWN / not-applicable rows usually have `[]` → gold span is `None` (nothing to cite).

This is a big deal: it means the faithfulness/abstention evaluation has real gold
evidence out of the box, without a separate manual annotation pass.

## If the upstream labels ever change

`mapToEviLabels` raises `KeyError` on any label it doesn't recognise — on purpose. If
that fires, the dataset's taxonomy changed; re-inspect `load_dataset(...).features` and
update the table above **and** the `_LABEL_MAP` dict in `ingest.py` together.

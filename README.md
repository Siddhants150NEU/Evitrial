# EVI-TRIAL

Evidence-verified clinical trial matching. Given a patient note, rank the trials the
patient might qualify for, and label every eligibility criterion `MET`, `NOT_MET`, or
`UNKNOWN` with a pointer to the exact sentence in the note that justifies the label.
If it can't point at real text, it abstains and says `UNKNOWN`.

The failure mode this targets: an LLM will confidently report that a patient meets
criterion X and cite a sentence that is not in the note. In a clinical setting, a
confident answer backed by a hallucinated reason is worse than "I don't know",
because it looks trustworthy.

Built on TrialGPT (NCBI/NIH, Nature Communications 2024) for the retrieve/match/rank
skeleton and the criterion annotations, MedCPT for biomedical dense retrieval, and
TREC Clinical Trials 2021 for the corpus and relevance judgments. The novelty is not
the skeleton. It is the verification and abstention layer wrapped around it.

## The one idea

Most RAG demos will hand you a confident answer with an invented citation. This
pipeline is built so it can't. Every criterion decision passes through one function,
`verify.verify()`. It checks that the cited span is literally present in the patient
note. If it isn't, the decision is forced to `UNKNOWN`. The eval scores a correct
label with a fabricated span as a failure, same as a wrong label.

The guarantee rests on three cooperating facts. `Decision.verified` starts `False` in
`schemas.py`. `verify.verify()` is the only code that sets it `True`.
`rank.aggregate()` asserts every decision is verified before anything gets scored.
Skip the verifier and `rank` raises. `tests/testContracts.py` proves both the forced
abstention and the assert. The whole enforcement mechanism is one tiny file and one
assert. Small surface, easy to trust.

Everything else in the repo (retrieval, matching, LoRA) supports that check. The gate
is the project.

## Quick start

```bash
# install
pip install -r requirements.txt

# sanity-check the data BEFORE building anything
python -m src.checkIngest        # --fast skips the slow 375k trial count

# run the invariants (pure python, no data needed)
python tests/testContracts.py    # or: pytest tests/
```

Run everything from the repo root; the `src` package lives there and imports directly,
no PYTHONPATH needed. A Makefile wraps the common commands (`make install`,
`make check`, `make test`, `make eval`, `make demo`), and `pip install -e .` works via
`pyproject.toml` if you want it installed.

`checkIngest` is a hard gate. Healthy output looks like this:

```
Trials (TREC CT 2021 corpus)
  [PASS]  corpus trial count               got 375580, expected 375580
Topics (synthetic patient notes)
  [PASS]  topic count                      got 75, expected 75
  [PASS]  every topic has note text        got True
Qrels (relevance judgements)
  [PASS]  total judgements                 got 35832, expected 35832
  [PASS]  relevance distribution (0/1/2)   got {0: 24243, 1: 6019, 2: 5570}, ...
Annotations (TrialGPT criterion labels -> matching eval)
  [PASS]  row count                        got 1015, expected 1015
  [PASS]  unique patients                  got 53, expected 53
  [PASS]  unique trials                    got 103, expected 103
  ...
ALL CHECKS PASSED -- ingestion looks healthy. Safe to build downstream.
```

If anything fails, stop and fix ingestion before building anything else. Every
downstream number is meaningless if the data loaded wrong.

## Data

Two datasets, two separate jobs. They are kept separate on purpose.

| Job | Dataset | Size | Scores |
|---|---|---|---|
| Retrieval | TREC CT 2021 (`clinicaltrials/2021/trec-ct-2021`) | 375,580 trials, 75 patient topics, 35,832 qrels | how well we find candidate trials |
| Matching | `ncbi/TrialGPT-Criterion-Annotations` | 1,015 pairs, 53 patients, 103 trials | MET/NOT_MET/UNKNOWN + evidence |

The TrialGPT set ships expert labels and expert evidence sentences, so the
faithfulness eval gets its gold spans without any extra annotation work.

## What goes where

Flat package, one file per pipeline stage. No `core/`, no `utils/`, no `models/`
maze. If you're not sure where code goes, the filename should answer it.

```
eviTrial/
  requirements.txt
  configs/default.yaml                  the one config, every knob lives here
  data/annotations/LABEL_MAPPING.md     why the six TrialGPT labels collapse to three
  reports/runs/                         one folder per eval run, written by eval.py
  notebooks/                            scratch and exploration
  tests/testContracts.py                the guardrails, 7 invariants
  src/                                  the package, one file per stage
```

Status markers: `done` = built and working, `harness` = harness works but the logic
is missing, `stub` = not built.

```
src/
  schemas.py       done     the dataclasses everything passes around (Trial, Criterion,
                            Candidate, Decision, TrialScore, PatientCriterionPair) plus
                            the single normalize(). Decision.verified defaults to False,
                            which is the linchpin of the gate
  config.py        done     loadConfig() reads default.yaml, setSeeds() pins the Python,
                            NumPy, and Torch RNGs. Boring on purpose
  trace.py         done     span() context manager, emits Langfuse spans if configured,
                            no-ops if not. Never crashes the pipeline
  pipeline.py      done     runPatient() wires every stage in order. Read this first
  ingest.py        done     loaders for trials, topics, qrels, and annotations, plus the
                            6->3 label mapping and gold-span extraction
  checkIngest.py   done     ingestion verifier, expected counts baked in, green/red
                            report, nonzero exit on failure
  verify.py        done     isSupported() substring check + verify(), the only setter of
                            verified=True. Nothing bypasses it
  rank.py          done     aggregate() turns decisions into one transparent linear
                            score and asserts every decision is verified
  retrieval.py     stub     the BM25 + MedCPT dense + rerank cascade, plus fetchTrials.
                            Never reads qrels
  indexQdrant.py   stub     one-time: embed the corpus into Qdrant
  parse.py         stub     eligibility text -> atomic Criterions, with negation and
                            temporal cues
  match.py         stub     match() dispatches ruleMatch / zeroShotMatch / loraMatch,
                            each returning a raw Decision (verified=False)
  trainLora.py     stub     one-time LoRA fine-tune, train split only
  eval.py          harness  runId, git SHA, seed, and JSON logging work; the six metric
                            functions do not exist yet
  demo.py          stub     minimal Streamlit UI, build last
```

## What happens on a query

`pipeline.runPatient(note, config)` is the whole runtime path. Read it and you
understand the system.

```
note
  retrieval.retrieve                        ->  [Candidate ...]
  retrieval.fetchTrials(ids)                ->  {nctId: Trial}
  for each candidate trial:
    parse.parseCriteria(trial)              ->  [Criterion ...]
    for each criterion:
      match.match(note, crit)               ->  Decision(verified=False)
      verify.verify(decision, note, crit.text)  ->  Decision(verified=True), or forced UNKNOWN
  rank.aggregate(nctId, decisions, score)   ->  TrialScore  (asserts every decision verified)
  sort by score                             ->  [TrialScore ...] + per-criterion evidence
                                                + missing-info list
```

Two offline jobs sit outside this path and run once. `indexQdrant.buildIndex` embeds
the whole corpus into Qdrant, a retrieval prerequisite. `trainLora.train` fits the
LoRA adapter, a matcher prerequisite if the LoRA rung is used.

Ingestion is the front door:

```
TREC CT 2021          ->  ingest.loadTrials / loadTopics / loadQrels  ->  Trial / topics / qrels
TrialGPT annotations  ->  ingest.loadAnnotations -> ingest.toEvalPairs  ->  [PatientCriterionPair]
                          checkIngest.py verifies the counts on both
```

## Commands

| Do this | Command |
|---|---|
| Install | `pip install -r requirements.txt` |
| Check the data | `python -m src.checkIngest` (`--fast` skips the full count) |
| Run guardrails | `python tests/testContracts.py` |
| Build the vector index (once, slow) | `python -m src.indexQdrant` |
| Fine-tune LoRA (once) | `python -m src.trainLora` |
| Run the full eval as a logged run | `python -m src.eval` |
| Demo | `streamlit run src/demo.py` |

## Stack

Committed choices, not placeholders. Swap only with a reason.

| Stage | Packages | Why |
|---|---|---|
| Ingestion | `ir_datasets`, `datasets` | canonical loaders for TREC CT and the HF annotations |
| Sparse retrieval | `bm25s` | fast, light BM25, less baggage than the alternatives |
| Dense retrieval | `transformers`, `torch`, `qdrant-client` | MedCPT encoders plus a real vector store |
| Encoders | MedCPT (`ncbi/MedCPT-*`) | domain-matched biomedical encoders |
| Parsing | stdlib + `re` | rules first, reach for a sentence splitter only if needed |
| Matching | `transformers`, `torch` | biomedical NLI classifier |
| Fine-tuning | `peft`, `accelerate` | LoRA adapters, cheap to train |
| Verification | stdlib only | the guarantee has to be simple to be trustworthy |
| Ranking | stdlib only | transparent linear formula |
| Metrics | `ir_measures`, `scikit-learn` | never hand-roll nDCG or F1 |
| Tracing | `langfuse` (optional) | per-stage spans, degrades to a no-op |
| Demo | `streamlit` | fastest path to a clickable UI |

One trap worth flagging: MedCPT is asymmetric. Patient notes go through the query
encoder, trials through the article encoder. Using one encoder for both is a classic
silent bug.

## How it's measured

Every number comes out of `python -m src.eval` and lands in `reports/runs/<runId>/`:
`config.json` (the exact config), `meta.json` (runId, git short SHA, seed, UTC
timestamp), and `metrics.json` (every metric). No run folder, no number.

Retrieval is scored against the TREC qrels with `ir_measures`: nDCG@10 primary,
Recall@10/20/50, MAP, one row per variant (BM25, dense, hybrid, hybrid plus rerank).
Matching is scored against the expert labels: macro-F1 primary, per-class
precision/recall/F1, confusion matrix, 95% bootstrap confidence intervals. Rungs are
compared on a test split that is frozen and split by patientId, so no patient appears
on both sides.

The headline is faithfulness. Supported-rate is the fraction of non-abstained
decisions whose cited span passes `verify.isSupported`. It gets reported with the
verifier ON and OFF, and the delta between the two is the project's contribution. The
forced-abstention count is reported next to it, since that is the mechanism behind
the delta.

Abstention gets its own numbers: coverage, selective accuracy on the answered subset,
UNKNOWN recall on the pairs where the experts themselves said there wasn't enough
information, and a risk-coverage curve. Calibration (ECE, Brier) and latency (p50/p95
plus the hardware it ran on) round it out, reported modestly given the sample size.

## Rules

1. camelCase for functions, variables, and filenames. PascalCase for classes.
2. One `normalize()`. One `default.yaml`. No forks, no magic numbers.
3. `verify()` is the only code that sets `verified=True`, and `rank` asserts it on
   every decision. Not bypassable.
4. Retrieval never reads the qrels. There is a test that enforces this.
5. No `runId`, no number. A metric that is not tied to a logged run does not exist.

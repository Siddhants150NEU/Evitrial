# 🧪 EVI-TRIAL

**Evidence-verified clinical-trial matching.** Given a patient note, rank the clinical
trials they might qualify for — and for every criterion, say **MET / NOT_MET / UNKNOWN**
*and point at the exact text that proves it*. If it can't point at real text, it shuts up
and says **UNKNOWN**.

> If you read one sentence, read this:
> **The point of this project is not retrieval. It's that every answer is backed by a
> real span or it abstains — checked in code, not vibes.**

---

## 🎯 The one idea (the "spine")

Most RAG demos will happily give you a confident answer with a made-up citation. This one
*can't*. There's a function — `verify.verify()` — that every decision must pass through.
If the cited span isn't literally in the patient note, the decision is **forced to
UNKNOWN**. A correct label with a fake span counts as a **failure**, not a win.

That's the whole differentiator. Everything else (retrieval, matching, LoRA) is
supporting cast.

```
A correct answer for invented reasons  ==  a wrong answer.
We refuse to ship either one.
```

---

## ⚡ Quick start (do this first)

```bash
# 1. install
pip install -r requirements.txt

# 2. sanity-check the data BEFORE building anything (this is a hard gate)
python -m src.checkIngest          # add --fast to skip the slow 375k trial count

# 3. run the invariants (no data needed; pure python)
python tests/testContracts.py           # or: pytest tests/
```

> ℹ️ The `src` package lives at the repo root, so run commands **from the repo root**
> and it imports directly — no `PYTHONPATH` needed. A `Makefile` wraps the common ones
> (`make install`, `make check`, `make test`, `make eval`, `make demo`), and `pip install -e .`
> works via `pyproject.toml` if you want it installed.

### ✅ What a healthy `checkIngest` looks like

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

If anything is red → **stop and fix ingestion.** Every number downstream is meaningless
if the data loaded wrong. (See `05_HARD_STOPS_AND_DEFENSIBILITY.md`.)

---

## 📊 What data are we even using?

Two datasets, two totally separate jobs. **Don't mix them** (this is on purpose).

| Job | Dataset | Size | Used for |
|---|---|---|---|
| **Retrieval** | TREC CT 2021 (`clinicaltrials/2021/trec-ct-2021`) | 375,580 trials · 75 patient topics · 35,832 qrels | scoring how well we *find* trials |
| **Matching** | `ncbi/TrialGPT-Criterion-Annotations` | 1,015 pairs · 53 patients · 103 trials | scoring MET/NOT_MET/UNKNOWN + evidence |

The matching set ships **expert labels AND expert evidence sentences**, so we get gold
spans for the faithfulness eval basically for free. 🎁 Full details:
`02_DATA_SPEC.md`.

---

## 🗺️ The map (what every file is, in one line)

Flat package, one file per pipeline stage. Colour key: 🟩 = built & working ·
🟨 = harness built, you fill logic · 🟥 = stub (your logic).

```
src/
  schemas.py       🟩 the vocabulary: dataclasses + the ONE normalize()
  config.py        🟩 load default.yaml + pin all the RNG seeds
  trace.py         🟩 optional Langfuse spans (no-ops if not configured)
  pipeline.py      🟩 runPatient(): the whole flow, top to bottom  ← read this first
  ingest.py        🟩 raw datasets -> clean schemas (WORKING)
  checkIngest.py   🟩 verify ingestion counts (WORKING, has expected numbers baked in)
  verify.py        🟩 THE SPINE: span check + forced abstention
  rank.py          🟩 transparent scoring + the "must be verified" gate
  retrieval.py     🟥 BM25 + MedCPT dense + rerank        (your logic)
  indexQdrant.py   🟥 one-time: embed trials into Qdrant   (your logic)
  parse.py         🟥 eligibility text -> atomic criteria  (your logic)
  match.py         🟥 MET/NOT_MET/UNKNOWN, 3 rungs         (your logic)
  trainLora.py     🟥 one-time: LoRA fine-tune            (your logic)
  eval.py          🟨 run harness WORKING; 6 metric fns are stubs (your logic)
  demo.py          🟥 minimal Streamlit UI (build LAST)

tests/testContracts.py   🟩 the guardrails (7 tests, all passing)
configs/default.yaml     🟩 every knob in one place
data/annotations/LABEL_MAPPING.md   🟩 the 6->3 label reasoning
```

---

## 🔁 The pipeline at a glance

```
   patient note
        │
        ▼
   [ retrieve ]  ── BM25 + MedCPT dense → fuse → rerank ──▶ top-k candidate trials
        │
        ▼   (for each trial)
   [ parse ]     ── eligibility blob → atomic criteria
        │
        ▼   (for each criterion)
   [ match ]     ── MET / NOT_MET / UNKNOWN  + a PROPOSED span   (verified=False)
        │
        ▼
   [ verify ]    ── span really in the note?  no → FORCE UNKNOWN.  sets verified=True   ★ the spine
        │
        ▼
   [ aggregate ] ── decisions → one transparent score   (asserts every decision verified)
        │
        ▼
   ranked trials + per-criterion evidence + "missing info" list
```

---

## 🧾 Command cheat-sheet

| Do this | Command |
|---|---|
| Install | `pip install -r requirements.txt` |
| Check the data | `python -m src.checkIngest` (`--fast` to skip full count) |
| Run guardrails | `python tests/testContracts.py` |
| Build the vector index (once, slow) | `python -m src.indexQdrant` |
| Fine-tune LoRA (once) | `python -m src.trainLora` |
| Run the full eval → logged run | `python -m src.eval` |
| Demo | `streamlit run src/demo.py` |

---

## 🚧 The 5 non-negotiables (the long version is in `CLAUDE.md`)

1. **camelCase everything** (functions, vars, filenames). Classes stay PascalCase.
2. **One `normalize()`**, one `default.yaml`. No forks, no magic numbers.
3. **`verify()` is non-bypassable.** Only it sets `verified=True`; `rank` asserts it.
4. **qrels are gold.** Retrieval never reads them. Ever. (There's a test for this.)
5. **No `runId`, no number.** Every result comes from a logged run or it doesn't exist.

---

## 📚 Where to go next

| I want to… | Open |
|---|---|
| Understand *why this exists* | `00_PROJECT_BRIEF.md` |
| See the architecture + stack | `01_ARCHITECTURE.md` |
| Know the exact data shapes + counts | `02_DATA_SPEC.md` |
| Know what to build, in what order | `03_IMPLEMENTATION_PLAN.md` |
| Know how it's measured | `04_EVALUATION_PROTOCOL.md` |
| Know the honesty rules + failure stops | `05_HARD_STOPS_AND_DEFENSIBILITY.md` |
| Look up a file/function's inputs & outputs | `CONTRACTS.md`  ← your day-to-day reference |
| Give the rules to Claude Code | `CLAUDE.md` |

---

## 📈 Status right now

- 🟩 **Working & tested here:** schemas, verify, rank, config, trace, pipeline wiring,
  ingest, checkIngest, the 7 contract tests. All compile; all invariants pass.
- 🟨 **Harness ready:** `eval.py` (run logging works; metrics are yours to fill).
- 🟥 **Your logic:** retrieval, indexQdrant, parse, match, trainLora, eval metrics, demo.
- ⚠️ **Not run here:** `ingest`/`checkIngest` need network + dataset downloads, so they
  were compiled and written against the *verified* dataset APIs but executed in *your*
  environment. First run may take a while (375k-trial corpus download + count).

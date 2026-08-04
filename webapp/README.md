# webapp/ — the demo

An interactive walkthrough of the pipeline. Ten tabs: one per pipeline stage, plus the
matcher registry, the results, and the supporting scripts. Type a patient note, run it,
then walk left to right watching **your** note move through every script.

Works two ways on purpose:

- **self-serve** — anyone clicks through at their own pace
- **presenter mode** — press `P`, everything scales up for a projector, arrow keys move

---

## Running it

```bash
python -m webapp.server.serveDemo
```

Then <http://localhost:8777>. That's cached-runs-only — instant, and it cannot fail
in front of an audience.

To let people run their own notes:

```bash
python -m webapp.server.serveDemo --live
```

| key | does |
|---|---|
| `←` `→` | move through tabs |
| `P` | presenter mode |
| `F` | fullscreen |

---

## Two pages worth calling out

**Start** is a single picker. Prebuilt runs and "write your own" are the same kind of
thing — options — and exactly one is selected at a time, so exactly one can run.

**Selecting anything reveals it, in the same slot.** Click a prebuilt note and you get
that patient's note read-only, sentence-indexed exactly as the results will cite it, plus
its facts as pills: rung, k, trials, criteria, how many times the gate fired, how long it
took, when it was built. Click "write your own" and the same slot becomes the editable
box with the rung buttons and `k`. Picking one always replaces the other, so what's on
screen is always what's about to run.

The button relabels itself too — *Load this run* vs *Run the pipeline*.

The note text rides along in the `/api/cache` listing rather than being fetched per
click, so switching between notes is instant instead of pulling a 350 KB run each time.

While a live run is going the whole picker disables, not just the button. The server
holds a lock and returns 429 to a second run, but nobody should have to find that out by
clicking.

**02 Retrieve** shows every trial that came back — full width, scrollable, one row each
with rank, title, condition, NCT id and its score components as pills. Above it, a funnel:
corpus → `topN` → rerank → `k`.

The metric pills are **not comparable to each other**, and the page says so. `hybrid` is
min-max normalised over the candidate set, which is why the bottom trial is always exactly
`0.000` — not a real zero. `rerank` is a raw cross-encoder logit, where negative means the
reranker doesn't believe the match. Worth having that sentence ready if someone asks why
your top hit scored zero.

How long the list is depends entirely on `k`. The default is now **10**, which gives a
list worth scrolling; `--k 3` if you want it short.

## Read this before you demo live

**A cold run takes about ten minutes.** Measured, not estimated — one note,
`sigir-20141`, rung `zeroShot`:

| stage | time |
|---|---|
| ingest | 0 ms |
| **retrieve** | **135,950 ms** |
| parse | 3 ms |
| **match** (126 criteria) | **496,712 ms** |
| verify | ~1 ms total (4.6 µs per decision, measured) |
| rank | under 1 ms |
| **total** | **632,669 ms — 10m 33s** |

Two separate problems, and they need different fixes.

### Retrieval: 136 s, mostly boot

Both causes are in `src/retrieval.py`:

1. [`bm25Search`](../src/retrieval.py) builds the BM25 index in memory over all 375,580
   trials on first call. It's cached in `_BM25` afterwards, so this is a slow **boot**,
   not a slow query. Unavoidable unless you persist the index.
2. [`fetchTrials`](../src/retrieval.py) re-streams **the entire corpus on every query**
   to pull out ~k trials. This one is a genuine per-query full scan, and it's the reason
   the second and third runs are still slow. It could reuse the trial list already sitting
   in `_BM25[0]`.

That second one is your logic, so it's flagged rather than fixed.

### Matching: 497 s, and this is the real wall

126 criteria took 497 seconds — **~3.7 s per criterion**, on MPS. (The app no longer
quotes this figure; it computes it from whatever runs are in `cache/`.) `zeroShotMatch` loops
over the note's sentences for every criterion, so cost is roughly
`k × criteria × sentences`. At k=10 that's sixteen minutes. At k=20 it's over an hour.

Batching the sentence loop into one forward pass per criterion is the obvious win, but
that's matcher logic, so it's your call.

**Keep `k` low for anything live.** k=20 is right for eval and wrong for a stage.

The practical answer: **pre-build your demo notes** (below), present from those, and keep
`--live` for a volunteer who genuinely doesn't mind waiting.

### One more thing worth knowing before you present

On the run currently in `cache/`, retrieval did badly. The patient is a 58-year-old woman
with chest pain and suspected angina. The three trials it returned:

| trial | title |
|---|---|
| NCT01599975 | Long-acting Methylphenidate (Concerta) vs. Placebo |
| NCT02088866 | Commercial Lidocaine Patch as a Treatment for Ear-ringing |
| NCT02099006 | Novel Topical Therapies for the Treatment of Genital Pain |

None of them are related to the presenting complaint, and the cross-encoder's own scores
say so — all strongly negative. This is one note, so it
isn't proof of anything general, and logged nDCG@10 for hybrid+rerank is 0.33 rather than
zero. But **don't present this particular cached run as a success story.** Build a couple
more with `runCache.py` and pick one where retrieval actually found something, or show
this one deliberately and talk about why the ranked scores all came out negative — which
is arguably the better talk.

Also cosmetic but confusing on stage: `Candidate.score` after reranking carries the
**hybrid** score, not the rerank score, and the lowest candidate is always exactly
`0.000` — that's min-max normalisation over the candidate set, not a real zero.

---

## Pre-building runs

```bash
python -m webapp.server.runCache --list
python -m webapp.server.runCache --patient sigir-20141 --k 10
```

Writes `webapp/cache/<patientId>__<rung>__k<k>.json`, which is byte-for-byte the shape a
live run returns — so the front end can't tell them apart, and neither can a bug.

Notes come from the **annotation set**, not the TREC topics, for two reasons: those notes
are already numbered (`0. … 1. …`), which is what `splitNumberedNote` expects; and they
carry expert labels, so the UI can show a `gold` chip beside the prediction wherever a
retrieved trial happens to be one of the annotated ones. Gold is **displayed only** —
nothing here writes a prediction back into an annotation.

Every patient gets its own file and its own card in the picker. Pass several `--patient`
flags to **one** invocation rather than launching a process each: the BM25 index and the
matcher weights are built once and shared. Measured across one process — first retrieve
135s, every retrieve after it 64s. That 64s is `fetchTrials` rescanning the corpus; the
71s difference is what the shared index buys you, per patient.

`k` is in the filename, so the same patient at k=3 and k=10 coexist rather than one
silently clobbering the other.

---

## Adding the generative matcher

This was the design constraint, so here is the whole procedure:

1. Write `generativeMatch(note, criterion, config) -> Decision` in `src/match.py`.
2. Add an `elif` for it in `match.match()`'s dispatch.
3. Restart the server.

That's it. It appears in the Matchers tab, in the run picker, and in every stage
readout. **No change to any file in `webapp/`.**

Why it works: `webapp/server/matcherRegistry.py` parses `match.py` with `ast` — it does
not import it — looking for functions named `<rung>Match`, rather than keeping its own
list. Reading instead of importing keeps `torch`, `transformers` and `peft` out of the
page load: ~4.3s and three `nltk.download` calls, on a screen where no model ever runs. To give the new rung a nicer
display name and blurb, add a row to `matcherRegistry.KNOWN` — there's already one
sitting there for `generative`, waiting.

Two things it handles for you:

- **Step 2 is optional, and there is a trap in it.** `match.match()` names `rules` and
  `zeroShot`, then ends in a bare `else: return loraMatch(...)`. The `raise
  NotImplementedError` on the line below it is unreachable. So `pipeline.runPatient(
  rung="lora")` does **not** raise — it works by falling into the `else` — and the day
  `generativeMatch()` lands without a dispatch edit, `rung="generative"` will hit that
  same `else` and return **LoRA output labelled generative**, silently, with no exception
  to catch.

  The demo defends against this: `runMatcher` only routes through `match.match()` for
  rungs `match()` names *explicitly*, and calls the function directly otherwise. The
  Matchers tab says which path each rung took and warns about the catch-all.

  **The real fix belongs in `match.py`, not here.** Replace the if/elif with a dict —
  `RUNGS = {"rules": ruleMatch, "zeroShot": zeroShotMatch, "lora": loraMatch}` and
  `return RUNGS[rung](...)`. A missing rung then raises `KeyError` instead of quietly
  returning the wrong model, and this whole caveat disappears.
- **A new `Decision` field flows through untouched.** If you add `rationale` for the
  generative rung, `decisionToDict` picks it up via `asdict` and the UI renders it under
  the criterion. No schema duplicated in JavaScript.

### What actually happened when it landed — read this, it changes the talk

The prediction above was **wrong**, and the reason is the interesting part.

`verify()` fired **zero** times on the generative rung too. Not because the model is
careful, but because `generativeMatch` can't produce an unsupported span *by
construction*:

```python
patientSpan = sentences[verdict.sentenceIndices[0]] if verdict.sentenceIndices else None
```

The LLM returns a sentence **index**, not a quotation. The span is then looked up from
the note's own sentences, so it is verbatim by definition. Checked empirically across
all three generative runs: every span present is a verbatim substring of its note. The
substring test cannot fail here, and no amount of hallucination would change that.

That is a *stronger* claim than the one the gate was built to make. Rather than checking
a quotation after the fact, `llmContract` makes an unfaithful span **unrepresentable**.
Constrain the output space instead of validating it afterwards.

**But abstention absolutely is happening — it just moved upstream.** Across the three
generative runs, `llmContract.checkVerdict` rejected 281 verdicts:

| patient | criteria | UNKNOWN | rejected by contract | model said UNKNOWN itself |
|---|---|---|---|---|
| sigir-20141 | 265 | 216 | 106 | 110 |
| sigir-20142 | 132 | 107 | 85 | 22 |
| sigir-20143 | 149 | 133 | 90 | 43 |

Rejection reasons, and they are worth reading out loud:

| reason | count | what it means |
|---|---|---|
| `wrongCriterion` | 120 | the model answered about a **different criterion** than the one asked |
| `emptyIndices` | 126 | claimed a label while citing no sentence at all |
| `emptyRationale` | 56 | no reasoning given |
| `schemaViolation` | 1 | malformed output |

`wrongCriterion` at ~19% of all calls is the one to chase. `buildPrompt` passes the
`criterionId` and `checkVerdict` requires it back; `qwen2.5:7b-instruct` echoes the wrong
one about a fifth of the time. That's prompt adherence, not eligibility reasoning, and it
is probably the cheapest accuracy win available right now.

**The honest framing for a talk:** two independent mechanisms produce abstention here —
a schema contract that rejects malformed or off-target verdicts, and a span check that is
structurally redundant for this rung but still guards the other three. Presenting
`verify()` as "the thing that catches the LLM" would be wrong. It never caught it,
because the interface never let it lie about spans.

---

## How it's put together

```
webapp/
  index.html              shell — tabs and mount points only
  appStyles.css           design tokens + every component
  appMain.js              all rendering; one `state.run` object drives every tab
  server/
    paths.py              where things live. Dependency-free on purpose (see below).
    serveDemo.py          stdlib http.server. Static files + JSON API, one origin.
    runStages.py          walks the pipeline emitting a per-stage event
    runCache.py           pre-computes runs into cache/
    matcherRegistry.py    discovers rungs by AST-parsing match.py
  cache/                  pre-computed runs (gitignored if they get large)
```

**No new dependencies.** The server is `http.server` from the standard library —
`fastapi` and `flask` aren't installed, and `starlette`/`uvicorn` are only here
transitively, which is not something to build on. Serving the API and the static files
from one origin also removes the CORS and `file://` headaches.

### Why `paths.py` exists for four lines

`serveDemo` and `runCache` both need the cache directory. Having `serveDemo` import it
from `runCache` looks tidier and costs 3.2 seconds of server startup, because `runCache`
→ `runStages` → `src.retrieval` → `torch`. A constants module with no imports keeps the
server's boot at 0.02s. Don't collapse it back.

### The one duplication, and why

`runStages.runNote()` walks the same modules in the same order as
`pipeline.runPatient()`. It doesn't reimplement any logic — every real decision is still
made by `src/` — but it *is* a second copy of the orchestration loop.

The alternative was adding an `onStage` callback to `pipeline.py`. That's the spine, and
editing it to serve a demo is the wrong trade. So: **if you change the stage order in
`pipeline.py`, change it here too.** That's the cost, stated plainly.

### API

| route | returns |
|---|---|
| `GET /api/matchers` | discovered rungs + whether live runs are enabled |
| `GET /api/cache` | list of pre-computed runs |
| `GET /api/cache/<id>` | one full run |
| `POST /api/run` | `{note, rung, k}` → a full run. 503 unless `--live`. |

Runs are serialised one at a time — two people hitting Run at once would thrash the
models and both get a bad experience, so the second gets a 429 and a "try again in a
moment".

---

## Design notes

Palette, type and motif come from the reference deck, not invented here:

| token | value | job |
|---|---|---|
| indigo | `#414370` | dominant — all body text, the score bubble |
| coral | `#F08486` | the one action colour. Buttons, the logo dot, accents |
| sky | `#96CCE1` / `#B8E1F1` | output panels, hover states |
| periwinkle | `#8AA2D4` | the hand-drawn squiggle rule |
| cream | `#FFFCF5` / `#FAF7F6` | grounds |

**Barlow** throughout (400–800), JetBrains Mono for evidence — spans, NCT ids, field
names, anything the machine said verbatim. The squiggle rule under each heading is an
inline SVG in `appStyles.css`, redrawn from the deck's doodle.

### The background doodles

18 hand-drawn medical icons drift behind everything at 16% opacity — capsules,
stethoscope, DNA, ambulance, microscope, heart-in-hands and so on. They're the
**vector originals** pulled out of the reference deck (`ppt/media/*.svg`, not the
PNGs), so they're crisp at any size and 156 KB for the whole set.

They live in `webapp/assets/icons/` and are placed by one block of markup at the top of
`index.html`. Each one carries four custom properties you can tune without touching CSS:

```html
<img src="assets/icons/dna.svg" alt="" style="--x:1%;--y:63%;--s:80px;--r:-8deg;--d:-7s">
<!--                                            where      how big   tilt      drift offset -->
```

Three rules keep them charming rather than annoying:

- **Content cards are opaque**, so a doodle behind a card is simply invisible. They only
  ever show through cream. That's why 16% is enough to read without competing.
- **Keep `y` between 18% and 40% clear across the middle** (`x` 10%–90%). That band is
  where the lede sits directly on cream, and it's the one place on any tab where a doodle
  would be behind *text* instead of behind a card. Two icons originally landed there and
  were moved out to the gutters.
- **They're decorative, and declared as such** — `aria-hidden`, empty `alt`,
  `pointer-events:none`. Screen readers never see them.

Motion is a single slow bob plus two degrees of sway, 19s, staggered by the `--d` offset.
Two axes of motion read as noise; one reads as alive. Under `prefers-reduced-motion` the
doodles stay and simply stop moving. In presenter mode they drop to 10% — on a projector
they wash out and start competing with the type, and legibility wins over charm.

At viewports under 820px the whole layer is hidden: there are no gutters at that width,
so every icon would land under body text.

**One licensing note:** these came out of a Canva template deck. Fine for your own talk
and this repo; if the demo ever gets deployed publicly, check what that template's
licence allows. Swapping them is a drop-in — replace the files in `assets/icons/`, keep
the names.

Status colours are **always** paired with the literal word (`MET` / `NOT_MET` /
`UNKNOWN`) and a border treatment — UNKNOWN is dashed everywhere. Never colour alone.
Beyond accessibility that's the honest reading: UNKNOWN is drawn as *held open*, not as
*failed*, because that's what it means.

Conventions match the repo: **camelCase** for files and functions (`appMain.js`,
`runStages.py`, `renderStage`), PascalCase for classes, standard names
(`index.html`, `README.md`) left alone.

---

## Known gaps

- **No streaming.** A live run is one POST that returns when it's finished; the spinner
  is a spinner, not progress. `runStages.runNote` already takes an `onStage` callback,
  so wiring it to SSE is a contained change if the generative rung makes runs long enough
  to need it.
- **The Numbers/eval tab isn't built yet.** The previous prototype had one reading from
  `reports/runs/`. Worth adding back, reading the run folder live rather than hardcoding.
- **Cached runs go stale silently.** If `parse.py` or a matcher changes, old cache files
  still load and still look authoritative. They record `builtAt` and the rung, but nothing
  checks them against the current code.

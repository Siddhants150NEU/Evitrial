/* ============================================================================
   EVI-TRIAL demo — front end.

   One idea holds the whole thing together: there is ONE run object in state, and
   every stage tab is a different window onto it. Load a cached run or compute a
   live one; the tabs don't know or care which, because both come back in the same
   shape from the server.

   Adding a matcher requires no change to this file. The rung buttons are rendered
   from /api/matchers, which reads match.py. That's the point.
   ========================================================================= */

const state = {
  run: null,          // the currently loaded run (cached or live)
  rungs: [],          // matcher rungs discovered on the server
  live: false,        // is the server accepting real runs?
  cached: [],         // list of pre-computed runs
  rung: "zeroShot",   // which rung the next live run should use
  k: 10,
  busy: false,
  presenting: false,
  selected: null,     // {type:"cached", id} | {type:"custom"} — exactly one, ever
  customNote: "",
};

const $ = (sel) => document.querySelector(sel);
const el = (html) => { const d = document.createElement("div"); d.innerHTML = html.trim(); return d.firstElementChild; };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* Stage metadata: what each script does, in plain language. This is the "walkthrough"
   half — it's true whether or not a run is loaded, so the tabs are never empty. */
const STAGES = [
  { id: "ingest", n: "01", title: "Ingest", script: "ingest.py", fn: "splitNumberedNote()",
    headline: "A patient arrives as prose.",
    what: "Before anything can be cited, the note has to become addressable. Each sentence gets an index, so later a decision can point at line 5 instead of gesturing at the whole note.",
    why: "Every downstream guarantee rests on this. A span you can't locate is a span you can't check." },
  { id: "retrieve", n: "02", title: "Retrieve", script: "retrieval.py", fn: "retrieve() + fetchTrials()",
    headline: "The whole corpus down to a handful.",
    what: "BM25 does a fast sparse pass, a dense MedCPT leg gets fused in, then a cross-encoder re-sorts what survives.",
    why: "This is a cascade, not full dense retrieval — the dense leg is fused in, it doesn't run the show. Retrieval never reads qrels; an AST test in testContracts.py enforces that." },
  { id: "parse", n: "03", title: "Parse", script: "parse.py", fn: "parseCriteria()",
    headline: "One blob becomes many questions.",
    what: "Eligibility text is a wall. Split it into atomic criteria — one testable rule each, typed inclusion or exclusion, with negation and temporal cues flagged.",
    why: "You can only be faithful about a claim small enough to actually check." },
  { id: "match", n: "04", title: "Match", script: "match.py", fn: "match()",
    headline: "The model gets to guess.",
    what: "For each criterion the active rung proposes a label plus the two spans it claims support that label: one from the trial text, one from the patient note.",
    why: "Nothing here is trusted. Every Decision comes back with verified=False. match() proposes; it never decides." },
  { id: "verify", n: "05", title: "Verify", script: "verify.py", fn: "verify()",
    headline: "The gate doesn't argue. It checks.",
    what: "normalize(patientSpan) in normalize(note), and normalize(trialSpan) in normalize(criterionText). Both true, or the label is rewritten to UNKNOWN.",
    why: "No confidence threshold, no second model — a substring test you can run in your head. This is the only place in the codebase where verified flips to True." },
  { id: "rank", n: "06", title: "Rank", script: "rank.py", fn: "aggregate()",
    headline: "A score you can read out loud.",
    what: "A linear sum with weights from configs/default.yaml. Inclusion met helps, inclusion failed hurts, an exclusion hit hurts five times as much.",
    why: "UNKNOWNs never move the score. They can't help a trial and they can't hurt it — they become missingInfo for a human to resolve. And aggregate() asserts every decision is verified, so skipping the gate raises rather than silently scoring." },
];

const SUPPORT = [
  { f: "schemas.py", role: "the vocabulary", p: "Every dataclass, plus normalize() — the one text cleaner, defined exactly once so the substring check in verify means the same thing everywhere.", io: "IN nothing · OUT Trial, Criterion, Candidate, Decision, TrialScore" },
  { f: "config.py", role: "settings + seeds", p: "Loads configs/default.yaml and pins the RNGs. If a number matters it lives in the YAML, never in code.", io: "IN default.yaml · OUT dict, seeded RNGs" },
  { f: "ingest.py", role: "datasets to schemas", p: "Streams the TREC corpus, the 75 topics, the 35,832 qrels, and the 1,015 expert criterion annotations. Maps the 6 expert labels down to our 3.", io: "IN dataset ids · OUT Trials, topics, qrels, PatientCriterionPairs" },
  { f: "checkIngest.py", role: "trust but verify", p: "Re-counts everything against known-good numbers and exits non-zero if a count drifted. Run it after touching ingestion.", io: "IN the datasets · OUT a printed report, exit code" },
  { f: "indexQdrant.py", role: "one-time embed", p: "Embeds every trial with the MedCPT article encoder into a persisted Qdrant collection. Runs once; the 2.9 GB store on disk is its output.", io: "IN config · OUT a Qdrant collection" },
  { f: "trainLora.py", role: "one-time fine-tune", p: "Fits a rank-16 LoRA adapter on the train split only. Never touches the frozen test split.", io: "IN train split · OUT an adapter on disk" },
  { f: "pipeline.py", role: "the flow", p: "retrieve, parse, match, verify, aggregate, sort. Thirty lines. Read it first if you want to understand the system.", io: "IN note, config · OUT list[TrialScore]" },
  { f: "eval.py", role: "the run harness", p: "Every reported number gets a runId and a folder under reports/runs/. If it wasn't logged there, it can't be quoted.", io: "IN config · OUT runId + metrics.json" },
  { f: "trace.py", role: "optional telemetry", p: "Wraps each stage in a Langfuse span if it's configured, and no-ops silently if it isn't. Never raises.", io: "IN a stage name · OUT a context manager" },
  { f: "testContracts.py", role: "the guardrails", p: "Seven tests that encode the promises: verify forces UNKNOWN on a missing span, rank refuses unverified decisions, retrieval never mentions qrels.", io: "IN the modules · OUT pass/fail" },
];

/* ---------- boot ---------------------------------------------------------- */
async function boot() {
  buildTabs();
  renderSupport();
  await loadMatchers();
  wireKeys();
  // No renderStart() here: show("start") routes through refreshStart(), which lists the
  // cache and renders. Doing both meant fetching /api/cache twice and building the Start
  // DOM twice on every load.
  show(location.hash.slice(1) || "start");
}

async function loadMatchers() {
  try {
    const r = await fetch("/api/matchers").then((x) => x.json());
    state.rungs = r.rungs || [];
    state.live = !!r.live;
    if (!state.rungs.some((x) => x.rung === state.rung) && state.rungs.length) {
      state.rung = state.rungs[state.rungs.length - 1].rung;
    }
  } catch { state.rungs = []; }
  renderMatchers();
  $("#livePill").className = "pill " + (state.live ? "live" : "off");
  $("#livePill").textContent = state.live ? "live runs ON" : "cached only";
}

async function loadCacheList() {
  try { state.cached = (await fetch("/api/cache").then((x) => x.json())).runs || []; }
  catch { state.cached = []; }
}

/* ---------- tabs ---------------------------------------------------------- */
const TABS = [
  { id: "start", label: "Start" },
  ...STAGES.map((s) => ({ id: s.id, label: `${s.n} ${s.title}` })),
  { id: "results", label: "Results" },
  { id: "matchers", label: "Matchers" },
  { id: "machine", label: "The rest" },
];

function buildTabs() {
  const nav = $("#tabs");
  TABS.forEach((t) => {
    const b = el(`<button class="tab" data-tab="${t.id}">${esc(t.label)}</button>`);
    b.onclick = () => show(t.id);
    nav.appendChild(b);
  });
  STAGES.forEach((s) => $("#views").appendChild(el(`<section class="view" id="view-${s.id}"></section>`)));
}

function show(id) {
  if (!TABS.some((t) => t.id === id)) id = "start";
  state.active = id;
  document.querySelectorAll(".view").forEach((v) => (v.dataset.active = v.id === `view-${id}`));
  document.querySelectorAll(".tab").forEach((b) => b.setAttribute("aria-current", b.dataset.tab === id));
  const stage = STAGES.find((s) => s.id === id);
  if (stage) renderStage(stage);
  if (id === "results") renderResults();
  // Re-render the picker on every visit: a cache file written while the page was open
  // should show up when you come back to Start, not only after a reload.
  if (id === "start") refreshStart();
  location.hash = id;
  window.scrollTo({ top: 0, behavior: "instant" });
}

function wireKeys() {
  addEventListener("keydown", (e) => {
    if (e.target.matches("textarea, input")) return;
    const i = TABS.findIndex((t) => t.id === state.active);
    if (e.key === "ArrowRight") { e.preventDefault(); show(TABS[Math.min(TABS.length - 1, i + 1)].id); }
    if (e.key === "ArrowLeft") { e.preventDefault(); show(TABS[Math.max(0, i - 1)].id); }
    if (e.key.toLowerCase() === "p") togglePresent();
    if (e.key.toLowerCase() === "f") {
      document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
    }
  });
  $("#presentBtn").onclick = togglePresent;
}

function togglePresent() {
  state.presenting = !state.presenting;
  document.body.classList.toggle("presenting", state.presenting);
  $("#presentBtn").setAttribute("aria-pressed", state.presenting);
}

/* ---------- start tab -----------------------------------------------------
   One picker. Prebuilt notes and your own are the same kind of thing — options —
   and exactly one of them is selected, so exactly one can run. The server also
   holds a lock, but the UI shouldn't let you get there in the first place.      */
function renderStart() {
  const busy = state.busy;
  const sel = state.selected;
  const isSel = (t, id) => sel && sel.type === t && (id === undefined || sel.id === id);

  const prebuilt = state.cached.map((r) => `
    <button class="pick" data-pick="cached" data-id="${esc(r.id)}"
            aria-pressed="${isSel("cached", r.id)}" ${busy ? "disabled" : ""}>
      <div class="ptop"><span class="pname">${esc(r.topicId || r.id)}</span>
        <span class="ptick">✓</span></div>
      <div class="pmeta">rung <b>${esc(r.rung)}</b> · k=${r.k} · ${r.trials} trials
        · ${r.forcedAbstentions} forced abstention${r.forcedAbstentions === 1 ? "" : "s"}</div>
    </button>`).join("");

  const noCache = `<div class="empty" style="grid-column:1/-1;padding:26px">
      <b>No prebuilt notes yet.</b><br>
      <code class="span" style="display:inline-block;margin-top:8px">python -m webapp.server.runCache --patient sigir-20141 --k 10</code>
    </div>`;

  $("#view-start").innerHTML = `
    <span class="eyebrow">Clinical trial matching · TREC-CT 2021</span>
    <h1>Every decision carries its receipt.</h1>
    <div class="squiggle"></div>
    <p class="lede">Most trial matchers are judged on whether they find the right trial. This one is
      judged on whether it can <b>show you the sentence</b> that justifies each call — and on whether
      it has the nerve to say <b>UNKNOWN</b> when it can't. Pick one note below, run it, then walk the
      tabs left to right to watch it move through every script in the pipeline.</p>

    <h2>Pick one note</h2>
    <p style="font-size:14px;color:var(--ink-dim);margin-bottom:14px">
      One at a time — the models are shared, so a second run would only slow the first one down.</p>

    <div class="pickGrid" style="margin-bottom:4px">
      ${state.cached.length ? prebuilt : noCache}
      <button class="pick writeOwn" data-pick="custom" aria-pressed="${isSel("custom")}"
              ${busy ? "disabled" : ""}>
        <div class="ptop"><span class="pname">Write your own</span><span class="ptick">✓</span></div>
        <div class="pmeta">Type a patient note and run the real pipeline over it.
          ${state.live ? "Slow, but real." : "Needs the server started with --live."}</div>
      </button>
    </div>

    <div id="customPane"></div>

    <div class="runBar">
      <span class="grow" id="runSummary"></span>
      <button class="btn" id="runBtn" ${busy ? "disabled" : ""}>Run</button>
    </div>
    <div id="runStatus" style="margin-top:14px"></div>

    <div class="grid g3" style="margin-top:26px">
      <div class="card tint-sky"><h3>match() proposes</h3>
        <p style="font-size:14px;color:var(--ink-dim);margin-top:6px">A label and two spans. Returns
          <code>verified=False</code>. It is allowed to be wrong.</p></div>
      <div class="card tint-coral"><h3>verify() checks</h3>
        <p style="font-size:14px;color:var(--ink-dim);margin-top:6px">Both spans real, or the label
          becomes UNKNOWN. The only place <code>verified</code> flips to True.</p></div>
      <div class="card"><h3>rank() refuses</h3>
        <p style="font-size:14px;color:var(--ink-dim);margin-top:6px"><code>assert all(d.verified)</code>.
          Break a link and the pipeline raises rather than scoring a guess.</p></div>
    </div>`;

  document.querySelectorAll("[data-pick]").forEach((b) => (b.onclick = () => {
    state.selected = b.dataset.pick === "custom"
      ? { type: "custom" } : { type: "cached", id: b.dataset.id };
    renderStart();
  }));
  renderCustomPane();
  renderRunBar();
  $("#runBtn").onclick = runSelected;
}

/* Selecting anything reveals it. A prebuilt note shows the note read-only; "write your
   own" shows the editable box. Same slot, so picking one always replaces the other. */
function renderCustomPane() {
  const pane = $("#customPane");
  if (!pane) return;
  if (!state.selected) { pane.innerHTML = ""; return; }
  if (state.selected.type === "cached") return renderCachedPane(pane);

  pane.innerHTML = `
    <div class="card" style="margin-top:14px">
      <h3>Your note</h3>
      <p style="font-size:13.5px;color:var(--ink-dim);margin:5px 0 11px">
        Numbered sentences work best — <code>0. …</code> <code>1. …</code> — because that's what the
        span check indexes against.</p>
      <textarea class="noteBox" id="noteBox" ${state.busy ? "disabled" : ""}
        placeholder="0. A 58-year-old woman presents to the ER with chest pain.&#10;1. She is known to have hypertension and obesity.&#10;2. She currently takes no medications.">${esc(state.customNote)}</textarea>
      <div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:13px">
        <div><div style="font-size:11px;color:var(--ink-soft);margin-bottom:5px">matcher rung</div>
          <div class="miniRungs">${state.rungs.map((r) => `
            <button class="miniRung" data-rung="${esc(r.rung)}"
              aria-pressed="${r.rung === state.rung}" ${state.busy ? "disabled" : ""}>${esc(r.label)}</button>`).join("")}</div></div>
        <div><div style="font-size:11px;color:var(--ink-soft);margin-bottom:5px">trials (k)</div>
          <input class="numBox" id="kBox" type="number" min="1" max="20" value="${state.k}"
            ${state.busy ? "disabled" : ""}></div>
      </div>
      <p style="font-size:12px;color:var(--unknown);margin-top:12px;line-height:1.5">
        Keep k small. ${liveCostEstimate()}</p>
    </div>`;

  $("#noteBox").oninput = (e) => { state.customNote = e.target.value; renderRunBar(); };
  pane.querySelectorAll("[data-rung]").forEach((b) => (b.onclick = () => {
    state.rung = b.dataset.rung; renderCustomPane(); renderRunBar();
  }));
  $("#kBox").onchange = (e) => {
    state.k = Math.max(1, Math.min(20, +e.target.value || 3)); renderRunBar();
  };
}

/* Estimate live-run cost from the runs actually on disk rather than quoting a frozen
   measurement. Hardcoded timings in UI copy go stale silently — these two already had. */
function liveCostEstimate() {
  const runs = state.cached.filter((r) => r.totalMs && r.criteria);
  if (!runs.length) return "No cached runs to estimate from yet — budget several minutes.";
  const ms = runs.reduce((a, r) => a + r.totalMs, 0);
  const crit = runs.reduce((a, r) => a + r.criteria, 0);
  const slowest = Math.max(...runs.map((r) => r.totalMs));
  return `Across ${runs.length} cached run${runs.length === 1 ? "" : "s"} that works out at
    <b>${(ms / crit / 1000).toFixed(1)}s per criterion</b>; the slowest took
    <b>${Math.round(slowest / 60000)}m ${Math.round(slowest % 60000 / 1000)}s</b> end to end.`;
}

function renderCachedPane(pane) {
  const r = state.cached.find((c) => c.id === state.selected.id);
  if (!r) { pane.innerHTML = ""; return; }

  const sents = r.sentences?.length ? sentencesHtml(r.sentences)
    : `<p style="color:var(--ink-soft)">This run predates the note being stored in the
         listing. Load it and the Ingest tab has it.</p>`;

  pane.innerHTML = `
    <div class="card" style="margin-top:14px">
      <div class="listHead">
        <h3>The note — ${esc(r.topicId || r.id)}</h3>
        <span class="pill">${r.sentences?.length || 0} sentences</span>
      </div>
      <p style="font-size:13.5px;color:var(--ink-dim);margin:5px 0 12px">
        Already run and cached, so loading it is instant. These indices are what every
        span in the results points back at.</p>
      <div class="note scroll" style="max-height:min(34vh,290px)">${sents}</div>
      <div class="metrics" style="margin-top:14px">
        ${metric("rung", r.rung, "primary")}
        ${metric("k", r.k)}
        ${metric("trials", r.trials)}
        ${metric("criteria", r.criteria ?? "—")}
        ${metric("gate fired", `${r.forcedAbstentions}×`, r.forcedAbstentions ? "warnMetric" : "")}
        ${r.totalMs ? metric("took", `${(r.totalMs / 1000).toFixed(0)}s`) : ""}
        ${r.builtAt ? metric("built", r.builtAt.replace("T", " ").replace("Z", "")) : ""}
      </div>
    </div>`;
}

function renderRunBar() {
  const s = $("#runSummary"), btn = $("#runBtn");
  if (!s || !btn) return;
  const sel = state.selected;
  const words = state.customNote.trim().length;

  // Work out what to say, then say it once. Four branches each setting the same three
  // properties was three chances for them to drift apart.
  const plan =
    !sel                       ? ["Nothing picked yet — choose a note above.", true, "Run"]
    : sel.type === "cached"    ? [`Ready to load <b>${esc(
        state.cached.find((c) => c.id === sel.id)?.topicId || sel.id)}</b> — prebuilt, opens instantly.`,
        state.busy, "Load this run"]
    : !state.live              ? ["<b>Live runs are off.</b> Restart the server with <code>--live</code> to run your own note.", true, "Run"]
    : !words                   ? ["Type a note and this button wakes up.", true, "Run"]
    : [`Ready to run <b>${esc(state.rung)}</b> over k=${state.k} trials. This is the real
        pipeline, so expect minutes rather than seconds.`, state.busy, "Run the pipeline"];

  [s.innerHTML, btn.disabled, btn.textContent] = plan;
}

/* Re-list the cache, then redraw. Keeps the current selection if it still exists —
   losing your pick because a background build finished would be obnoxious. */
async function refreshStart() {
  if (state.busy) return;
  await loadCacheList();
  if (state.selected?.type === "cached" &&
      !state.cached.some((c) => c.id === state.selected.id)) {
    state.selected = null;
  }
  renderStart();
}

function runSelected() {
  if (state.busy || !state.selected) return;
  return state.selected.type === "cached" ? loadCached(state.selected.id) : runLive();
}


async function loadCached(id) {
  setStatus(`<div class="span">loading ${esc(id)} …</div>`);
  try {
    state.run = await fetch(`/api/cache/${encodeURIComponent(id)}`).then((r) => r.json());
    if (state.run.error) throw new Error(state.run.error);
    setStatus(`<div class="span ok"><b>Loaded.</b> ${state.run.trials.length} trials ·
      ${state.run.forcedAbstentions} forced abstentions · rung <b>${esc(state.run.rung)}</b>.
      Walk the tabs, or jump to Results.</div>`);
    show("ingest");
  } catch (err) { setStatus(`<div class="span bad">Couldn't load it: ${esc(err.message)}</div>`); }
}

async function runLive() {
  const note = state.customNote.trim();
  if (!note) return setStatus(`<div class="span bad">Type a note first.</div>`);
  if (state.busy) return;

  // Lock the whole picker, not just the button. The server refuses a second run with a
  // 429 anyway, but nobody should have to discover that by clicking.
  state.busy = true;
  renderStart();
  setStatus(`<div class="empty" style="padding:26px"><div class="spinner"></div>
    Running <b>${esc(state.rung)}</b> over k=${state.k} trials.<br>
    <span style="font-size:12.5px">On a cold server the BM25 index gets built first, which
    dominates. ${liveCostEstimate()}
    Leave it be; it can't be hurried and a second run would only make it worse.</span></div>`);
  try {
    const res = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ note, rung: state.rung, k: state.k }),
    }).then((r) => r.json());
    if (res.error) throw new Error(res.error);
    state.run = res;
    state.busy = false;
    renderStart();
    setStatus(`<div class="span ok"><b>Done in ${(res.totalMs / 1000).toFixed(1)}s.</b>
      ${res.trials.length} trials · ${res.forcedAbstentions} forced abstentions.</div>`);
    show("ingest");
  } catch (err) {
    state.busy = false;
    renderStart();
    setStatus(`<div class="span bad"><b>The run failed.</b> ${esc(err.message)}
      <br><span style="font-size:11px">Shown as-is rather than swallowed — a demo that hides errors
      isn't demonstrating much.</span></div>`);
  }
}

const setStatus = (html) => { const n = $("#runStatus"); if (n) n.innerHTML = html; };

/* ---------- stage tabs ---------------------------------------------------- */
function renderStage(stage) {
  const view = $(`#view-${stage.id}`);
  const ev = state.run?.stages?.find((s) => s.id === stage.id);

  view.innerHTML = `
    <span class="eyebrow">Stage ${stage.n} · ${esc(stage.script)}</span>
    <h1>${esc(stage.headline)}</h1>
    <div class="squiggle"></div>
    <div class="grid g2" style="margin-bottom:20px">
      <div class="card"><h3>What this script does</h3>
        <p style="font-size:15px;color:var(--ink-dim);margin-top:7px">${esc(stage.what)}</p></div>
      <div class="card tint-coral"><h3>Why it's there</h3>
        <p style="font-size:15px;color:var(--ink-dim);margin-top:7px">${esc(stage.why)}</p></div>
    </div>
    <div id="io-${stage.id}"></div>`;

  const slot = $(`#io-${stage.id}`);
  if (!ev) {
    slot.innerHTML = `<div class="empty"><b>No run loaded yet.</b><br>
      Head back to <b>Start</b>, pick a cached note or type your own, and this panel fills in
      with what actually went in and came out.</div>`;
    return;
  }
  slot.appendChild(el(`
    <div class="io">
      <div class="ioPanel"><div class="ioHead"><span class="bub"></span>Input</div>
        <div id="in-${stage.id}"></div></div>
      <div class="ioMid"><span class="fn">${esc(stage.fn)}</span><span class="arrow">→</span>
        <span class="ms">${ev.ms} ms</span></div>
      <div class="ioPanel out"><div class="ioHead"><span class="bub"></span>Output</div>
        <div id="out-${stage.id}"></div></div>
    </div>`));
  // Fall back to a generic dump rather than throwing: a stage added to runStages.py
  // before it has a bespoke renderer should still show its real data.
  const io = { ingest: ioIngest, retrieve: ioRetrieve, parse: ioParse,
               match: ioMatch, verify: ioVerify, rank: ioRank }[stage.id] || ioGeneric;
  io(ev, stage.id);
}

/* One metric chip. Was hand-written 11 times across renderCachedPane and retrievedRow. */
const metric = (k, v, cls = "") =>
  `<div class="metric ${cls}"><span class="mk">${esc(k)}</span><span class="mv">${esc(v)}</span></div>`;

/* Fallback stage renderer: real data, generic framing. Beats a dead tab. */
function ioGeneric(ev, id) {
  $(`#in-${id}`).innerHTML = kvList(ev.in || {});
  $(`#out-${id}`).innerHTML = kvList(ev.out || {});
}

const kvList = (obj) => Object.entries(obj)
  .map(([k, v]) => `<div class="kv"><span class="k">${esc(k)}</span><span class="v">${esc(
    typeof v === "object" ? JSON.stringify(v) : v)}</span></div>`).join("");

function ioIngest(ev, id) {
  $(`#in-${id}`).innerHTML = `<div class="span" style="max-height:300px;overflow:auto">${esc(ev.in.note)}</div>`;
  $(`#out-${id}`).innerHTML = `<div class="kv"><span class="k">sentences</span>
    <span class="v">${ev.out.count}</span></div>
    <div class="note scroll" style="margin-top:10px">${sentencesHtml(ev.out.sentences)}</div>`;
}

const sentencesHtml = (sents) => sents.map((t, i) =>
  `<span class="sent"><span class="idx">${i}</span>${esc(t)}</span>`).join("");

function ioRetrieve(ev, id) {
  const cands = ev.out.candidates;
  const cfg = ev.in;

  $(`#in-${id}`).innerHTML = kvList(cfg);
  $(`#out-${id}`).innerHTML = `
    <div class="kv"><span class="k">returned</span><span class="v">${cands.length}</span></div>
    <div class="kv"><span class="k">resolved to a trial</span>
      <span class="v">${ev.out.fetched ?? cands.length}</span></div>
    <div class="funnel">
      <div class="fstep"><b>${cfg.corpus ? cfg.corpus.toLocaleString() : "—"}</b><span>corpus</span></div>
      <span class="ftick">→</span>
      <div class="fstep"><b>${cfg.topN ?? "—"}</b><span>bm25 topN</span></div>
      <span class="ftick">→</span>
      <div class="fstep"><b>${cfg.useRerank ? "rerank" : "no rerank"}</b><span>cross-enc</span></div>
      <span class="ftick">→</span>
      <div class="fstep now"><b>${cands.length}</b><span>k returned</span></div>
    </div>`;

  // The candidate list gets the full width, below the IN→OUT row. Three columns is
  // fine for a summary and useless for a list you actually want to read.
  const host = $(`#io-${id}`);
  host.appendChild(el(`
    <div class="card" style="margin-top:16px">
      <div class="listHead">
        <h2 style="margin:0">Every trial retrieved</h2>
        <span class="pill">${cands.length} of k=${cfg.k}</span>
      </div>
      <p style="font-size:13.5px;color:var(--ink-dim);margin:6px 0 14px">
        Ordered as retrieval ranked them. <b>Scores are not comparable across columns</b> —
        hybrid is normalised 0–1 over this candidate set, so the bottom one is always
        exactly 0.000; rerank is a raw cross-encoder logit, where negative means the
        reranker doesn't believe in the match.</p>
      <div class="trialList scroll" id="retList"></div>
    </div>`));

  $("#retList").innerHTML = cands.map((c) => retrievedRow(c)).join("");
}

/* One row per retrieved trial. Falls back to run.trials for the title when the run was
   cached before the server started attaching metadata to candidates. */
function retrievedRow(c) {
  const scored = (state.run?.trials || []).find((t) => t.nctId === c.nctId);
  const title = c.title || scored?.title || "(title not captured in this run)";
  const condition = c.condition || scored?.condition;
  const bd = c.retrieverBreakdown || {};
  const metrics = Object.entries(bd)
    .map(([k, v]) => metric(k, typeof v === "number" ? v.toFixed(3) : v)).join("");

  return `<div class="retRow">
    <div class="retRank">${c.rank ?? "·"}</div>
    <div class="retBody">
      <h3>${esc(title)}</h3>
      <div class="retMeta">${esc(c.nctId)}${condition ? " · " + esc(condition) : ""}</div>
      ${c.summary ? `<p class="retSummary">${esc(c.summary)}</p>` : ""}
      <div class="metrics">
        ${metric("score", c.score.toFixed(3), "primary")}
        ${metrics}
        ${scored ? metric("criteria", scored.rows.length) : ""}
        ${c.fetched === false ? metric("fetchTrials", "missed", "warnMetric") : ""}
      </div>
    </div>
  </div>`;
}

function ioParse(ev, id) {
  $(`#in-${id}`).innerHTML = kvList(ev.in);
  const all = ev.out.perTrial;
  $(`#out-${id}`).innerHTML = `<div class="kv"><span class="k">criteria total</span>
    <span class="v">${ev.out.total}</span></div>
    <div class="stack scroll" style="margin-top:10px">${all.map((t) => `
      <div><div style="font-size:12px;font-family:var(--mono);color:var(--ink-soft);margin-bottom:5px">
        ${esc(t.nctId)} · ${t.criteria.length} criteria</div>
        ${t.criteria.slice(0, 6).map((c) => `<div class="span" style="margin-bottom:5px">
          <span class="lab ${c.criterionType === "inclusion" ? "inc" : "exc"}">${c.criterionType === "inclusion" ? "INC" : "EXC"}</span>
          ${esc(c.text.slice(0, 120))}</div>`).join("")}
        ${t.criteria.length > 6 ? `<div class="span" style="color:var(--ink-soft)">… ${t.criteria.length - 6} more</div>` : ""}
      </div>`).join("")}</div>`;
}

function ioMatch(ev, id) {
  $(`#in-${id}`).innerHTML = kvList(ev.in);
  const rows = allRows().slice(0, 8);
  $(`#out-${id}`).innerHTML = `${kvList(ev.out.labels)}
    <div style="font-size:11.5px;color:var(--ink-soft);margin:10px 0 6px">
      raw proposals, verified=False</div>
    <div class="stack scroll">${rows.map((r) => `
      <div class="span"><span class="lab ${r.raw.label}">${r.raw.label}</span>
        <span style="margin-left:6px">${esc(r.criterion.text.slice(0, 90))}</span>
        <div style="margin-top:5px;color:var(--ink-soft)">claims: “${esc((r.raw.patientSpan || "—").slice(0, 100))}”</div>
      </div>`).join("")}</div>`;
}

function ioVerify(ev, id) {
  const forced = allRows().filter((r) => r.forced);
  $(`#in-${id}`).innerHTML = kvList(ev.in) + `
    <div style="margin-top:12px" class="span">normalize(patientSpan) in normalize(note)<br>
      &nbsp;&nbsp;and normalize(trialSpan) in normalize(criterionText)</div>`;
  $(`#out-${id}`).innerHTML = `
    <div class="kv"><span class="k">forced abstentions</span>
      <span class="v" style="color:var(--unknown)">${ev.out.forcedAbstentions}</span></div>
    <div class="kv"><span class="k">supported rate</span>
      <span class="v">${ev.out.supportedRate ?? "—"}</span></div>
    ${kvList(ev.out.labels)}
    <div style="margin-top:12px">${forced.length ? `
      <div style="font-size:11.5px;color:var(--unknown);margin-bottom:7px">
        the gate rewrote these:</div>
      <div class="stack scroll">${forced.map((r) => `
        <div class="span bad">${esc(r.criterion.text.slice(0, 90))}<br>
          <span class="lab ${r.raw.label}">${r.raw.label}</span> →
          <span class="lab UNKNOWN">UNKNOWN</span><br>
          <s>“${esc((r.raw.patientSpan || "").slice(0, 90))}”</s> not in the note</div>`).join("")}</div>`
      : `<div class="span"><b>The gate fired zero times on this run.</b><br>
          <span style="color:var(--ink-soft)">That isn't proof it's unnecessary — this rung copies spans
          verbatim, so there was nothing to catch. The honest claim is narrow: the guarantee held and
          cost nothing. A paraphrasing matcher is where this gets interesting.</span></div>`}
    </div>`;
}

function ioRank(ev, id) {
  $(`#in-${id}`).innerHTML = kvList(ev.in.weights) + `
    <div class="span" style="margin-top:12px">assert all(d.verified for d in decisions)<br>
      <span style="color:var(--ink-soft)">↳ raises if verify() was skipped</span></div>`;
  $(`#out-${id}`).innerHTML = `<div class="stack">${ev.out.ranked.map((r, i) => `
    <div class="span ${i === 0 ? "ok" : ""}"><b>${esc(r.nctId)}</b> · score ${r.score.toFixed(3)}
      · ${r.missing} unknown${r.missing === 1 ? "" : "s"} parked as missingInfo</div>`).join("")}</div>`;
}

const allRows = () => (state.run?.trials || []).flatMap((t) => t.rows);

/* ---------- results ------------------------------------------------------- */
function renderResults() {
  const v = $("#view-results");
  if (!state.run) {
    v.innerHTML = `<h1>Results</h1><div class="empty"><b>Nothing to show yet.</b><br>
      Load or run a note on the Start tab.</div>`;
    return;
  }
  const run = state.run;
  const rows = allRows();
  const tally = (lab) => rows.filter((r) => r.verified.label === lab).length;
  const gold = run.gold || {};

  v.innerHTML = `
    <span class="eyebrow">${esc(run.source || "run")} · rung ${esc(run.rung)} · k=${run.k}</span>
    <h1>Ranked trials</h1>
    <div class="squiggle"></div>
    <div class="grid g3" style="margin-bottom:20px">
      <div class="card stat indigo"><div class="big">${run.trials.length}</div>
        <div class="lbl">trials scored</div></div>
      <div class="card stat unknown"><div class="big">${run.forcedAbstentions}</div>
        <div class="lbl">decisions the gate rewrote to UNKNOWN</div></div>
      <div class="card stat met"><div class="big">${(run.totalMs / 1000).toFixed(1)}s</div>
        <div class="lbl">end to end${run.source === "cached" ? ", when it was built" : ""}</div></div>
    </div>
    <div class="grid g3" style="margin-bottom:24px">
      <div class="card stat met"><div class="big">${tally("MET")}</div><div class="lbl">MET</div></div>
      <div class="card stat coral"><div class="big">${tally("NOT_MET")}</div><div class="lbl">NOT_MET</div></div>
      <div class="card stat unknown"><div class="big">${tally("UNKNOWN")}</div><div class="lbl">UNKNOWN — parked, not guessed</div></div>
    </div>
    ${run.trials.map((t) => trialCard(t, gold[t.nctId])).join("")}`;
}

function trialCard(t, goldForTrial) {
  const inc = t.rows.filter((r) => r.criterion.criterionType === "inclusion");
  const exc = t.rows.filter((r) => r.criterion.criterionType === "exclusion");
  const count = (rs, lab) => rs.filter((r) => r.verified.label === lab).length;
  return `<div class="trialCard">
    <div class="trialTop">
      <div class="scoreBub"><div><div class="n">${t.score.toFixed(2)}</div><div class="l">SCORE</div></div></div>
      <div class="meta">
        <h3>${esc(t.title || t.nctId)}</h3>
        <div class="nct">${esc(t.nctId)} · retrieval ${t.retrievalScore.toFixed(3)}
          · matched via ${esc(t.calledVia || "—")}</div>
        <div class="tallies">
          <span class="lab inc">${inc.length} inclusion</span>
          <span class="lab exc">${exc.length} exclusion</span>
          <span class="lab MET">${count(t.rows, "MET")} MET</span>
          <span class="lab NOT_MET">${count(t.rows, "NOT_MET")} NOT_MET</span>
          <span class="lab UNKNOWN">${count(t.rows, "UNKNOWN")} UNKNOWN</span>
        </div>
      </div>
    </div>
    ${t.rows.map((r) => critRow(r, goldForTrial)).join("")}
    ${t.missingInfo.length ? `<div class="span" style="margin-top:12px">
      <b>missingInfo</b> — ${t.missingInfo.length} question${t.missingInfo.length === 1 ? "" : "s"}
      a human still has to answer before this trial can be ruled in or out.</div>` : ""}
  </div>`;
}

function critRow(r, goldForTrial) {
  const g = goldForTrial?.[r.criterion.text];
  return `<div class="critRow ${r.forced ? "wasForced" : ""}">
    <span class="lab ${r.verified.label}">${r.verified.label}</span>
    <div class="text">${esc(r.criterion.text)}
      ${r.forced ? `<span class="forcedTag">gate rewrote this: ${esc(r.raw.label)} → UNKNOWN,
        claimed span wasn't in the note</span>` : ""}
      ${r.verified.patientSpan ? `<div class="span" style="margin-top:5px">“${esc(r.verified.patientSpan)}”</div>` : ""}
      ${renderExtra(r.verified.extra)}
    </div>
    ${g ? `<span class="lab gold" title="expert annotation">gold ${esc(g)}</span>` : "<span></span>"}
  </div>`;
}

/* Anything the Decision dataclass grows later — a `rationale` from the generative
   rung, say — lands here automatically instead of being dropped on the floor. */
function renderExtra(extra) {
  if (!extra) return "";
  // Drop empties: an absent `failures` shouldn't draw an empty box on every row.
  const rows = Object.entries(extra).filter(([, v]) =>
    Array.isArray(v) ? v.length : v !== null && v !== undefined && v !== "");
  if (!rows.length) return "";

  return rows.map(([k, v]) => {
    // A list of strings is a list, not a JSON blob. `failures` from the generative
    // rung is the interesting case: it's the llmContract check explaining exactly why
    // a verdict was rejected, which is the most quotable output in the whole run.
    if (Array.isArray(v) && v.every((x) => typeof x === "string")) {
      const bad = k === "failures";
      return `<div class="span ${bad ? "bad" : ""}" style="margin-top:5px">
        <b>${esc(bad ? "contract failures" : k)}</b>
        ${v.map((x) => `<div style="margin-top:3px">· ${esc(x)}</div>`).join("")}</div>`;
    }
    return `<div class="span" style="margin-top:5px;background:var(--sky-wash)">
      <b>${esc(k)}</b> ${esc(typeof v === "object" ? JSON.stringify(v) : v)}</div>`;
  }).join("");
}

/* ---------- matchers tab -------------------------------------------------- */
function renderMatchers() {
  const cards = state.rungs.map((r) => `
    <button class="rung" data-rung="${esc(r.rung)}" aria-pressed="${r.rung === state.rung}">
      <div class="top"><span class="name">${esc(r.label)}</span><span class="tag">${esc(r.tag)}</span></div>
      <p class="blurb">${esc(r.blurb)}</p>
      <div class="span" style="margin-top:9px;font-size:10.5px">${esc(r.fnName)}()</div>
      ${r.dispatched ? "" : `<div class="warn">match.match() doesn't name this rung.
        ${r.catchAll ? `Worse, it ends in a bare <code>else</code>, so asking it for
          &ldquo;${esc(r.rung)}&rdquo; would silently return a <em>different</em> matcher's output
          under this name. The demo sidesteps that and calls ${esc(r.fnName)}() directly.`
        : `The demo calls ${esc(r.fnName)}() directly.`}
        Add an explicit branch to the dispatch and this note disappears.</div>`}
    </button>`).join("");

  const ghost = state.rungs.some((r) => r.rung === "generative") ? "" : `
    <div class="rung ghostSlot">
      <div class="top"><span class="name">Generative</span><span class="tag">not yet</span></div>
      <p class="blurb">Write <code>generativeMatch()</code> in match.py and restart the server.
        It shows up here on its own — this UI reads match.py, it doesn't keep its own list.
        A <code>rationale</code> field on Decision would render too, with no change here.</p>
    </div>`;

  const v = $("#view-matchers");
  if (!v) return;
  v.innerHTML = `
    <span class="eyebrow">match.py · the rungs</span>
    <h1>Swap the matcher, keep the guarantee.</h1>
    <div class="squiggle"></div>
    <p class="lede">Every rung has the same signature — <code>match(note, criterion, config) → Decision</code>
      — so they're interchangeable and directly comparable. Whichever one you pick,
      <b>verify() still stands between it and the score.</b> A better matcher earns a better number;
      it never earns the right to skip the check.</p>
    <div class="rungGrid" style="margin-bottom:22px">${cards}${ghost}</div>
    <div class="card tint-cream">
      <h3>How this list is built</h3>
      <p style="font-size:14px;color:var(--ink-dim);margin-top:7px">
        The server parses <code>src/match.py</code> with <code>ast</code> — it doesn't import it —
        looking for functions named <code>&lt;rung&gt;Match</code>. Nothing about the matchers is
        hardcoded in the front end, so adding one is a Python change only. Give it a nicer label by
        adding a row to <code>matcherRegistry.KNOWN</code>.</p>
      <p style="font-size:13px;color:var(--ink-dim);margin-top:9px">
        Reading rather than importing keeps <code>torch</code>, <code>transformers</code> and
        <code>peft</code> out of the page load — about four seconds, on a screen where no model
        is going to run.</p>
    </div>`;
  v.querySelectorAll("[data-rung]").forEach((b) => (b.onclick = () => {
    state.rung = b.dataset.rung;
    renderMatchers();
  }));
}

/* ---------- the rest of the machine --------------------------------------- */
function renderSupport() {
  $("#view-machine").innerHTML = `
    <span class="eyebrow">Everything else in src/</span>
    <h1>The rest of the machine.</h1>
    <div class="squiggle"></div>
    <p class="lede">The six stages get their own tabs because they're the story. These are the
      scripts holding them up — the vocabulary, the data loading, the training, and the tests that
      stop any of it from quietly drifting.</p>
    <div class="grid g3">${SUPPORT.map((s) => `
      <div class="scriptCard"><div class="fname">${esc(s.f)}</div>
        <div class="role">${esc(s.role)}</div><p>${esc(s.p)}</p>
        <div class="io2">${esc(s.io).replace(/IN /, "<b>IN</b> ").replace(/ · OUT /, " · <b>OUT</b> ")}</div>
      </div>`).join("")}</div>`;
}

boot();

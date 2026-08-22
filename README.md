# ResumeFit AI — JD-to-Resume Optimization Engine

Turn every job description into a truthful, ATS-optimized resume.

You give it your **master resume** and a **job description**. It produces a
tailored resume built only from things you have actually done, plus an explainable
ATS score, a requirement-by-requirement evidence matrix, a gap report, and a
diff explaining every change.

---

## No API key. No network. No model.

The default engine is **pure rules**. It parses your resume, decomposes the JD,
matches requirements against evidence, scores, validates and writes the tailored
document entirely in Python. It runs in about a second and never makes a network
call.

That is not a downgrade — it makes the core guarantee *stronger*:

> The writer never composes a factual sentence. It **selects, ranks, reorders and
> reformats your own text.** There is no generative step that could invent a
> metric, so fabrication is structurally impossible rather than merely prohibited.

The summary is the single place prose is assembled, and it is a template whose
every slot is filled from a parsed field — any slot that cannot be filled is
dropped rather than guessed.

| Engine (`LLM_PROVIDER`) | Key | Network | What it adds |
|---|---|---|---|
| **`local`** (default) | none | none | Everything below. Bullets kept verbatim. |
| `ollama` | none | none | A local model rewords bullets; rejected if it adds a number |
| `anthropic` / `openai` / `gemini` | yes | yes | Hosted model rewording |

The hosted providers are optional and inert unless you switch to them.

## The one thing that makes this different

Most resume tools are keyword stuffers with a language model attached. This one
answers a harder question:

> **How strongly does this candidate's real experience support the requirements
> of this specific job?**

That's enforced structurally. The LLM does *extraction and writing*. Python does
*matching, scoring, and validation*. Every number the UI shows is computed in code
from inspectable inputs, and a deterministic **truthfulness gate** blocks any
document containing a fact that isn't in your master resume.

That gate is not advisory. If the generator invents a metric, the run is marked
`needs_review` and the offending strings are shown to you.

```
$ pytest tests/test_validators.py -q
...  invented percentage         → caught
     invented scale              → caught
     inflated team size          → caught
     unsupported technology      → caught
     altered company name        → caught
     re-levelled job title       → caught
     shifted employment dates    → caught
     added certification         → caught
     overclaimed years           → caught
     invented visa/clearance     → caught
     honest rewording            → passes
```

### On "100% ATS"

The generator reaches **100/100 on format compliance by construction** — it
controls every byte of output, so it emits standard headings, one date format,
safe glyphs, no tables, no images, no text boxes, and contact details in the
body. That number is real and reproducible.

What nobody can promise is a *universal* 100%, and it is worth knowing why:
independent testing found two-column layouts scrambled in 7 of 8 ATS products
and tables dropping content in 5 of 8, while field-level parsing accuracy tops
out near 87% even on clean documents. The same resume fed to Workday,
Greenhouse, Lever, iCIMS and Taleo yields different numbers of extracted skills
and jobs. So: 100% on the rules that are safe everywhere, and an honest account
of the rest.

(The widely repeated "75% of resumes are rejected by an ATS" figure has no study
behind it — it traces to a 2012 vendor sales pitch. This project does not use it.)

**Download DOCX unless the posting asks for PDF.** Modern cloud parsers handle
both, but legacy on-premise Taleo configurations still fail on a share of PDFs
due to an older text-extraction library. The UI marks DOCX as the default for
that reason.

| Score | Band |
|---|---|
| 95–100 | Excellent |
| 90–94 | Strong |
| 80–89 | Good |
| 70–79 | Needs improvement |
| < 70 | Poor alignment |

---

## Quick start

```bash
git clone <this repo> && cd resume-fit-ai
./run.sh          # → http://127.0.0.1:8000
```

That is the whole setup. No key, no signup, no model download.

Upload your master resume (PDF, DOCX, **HTML**, TXT or Markdown), paste the job
description, click Analyze, then Generate.

Manual setup:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your key
uvicorn backend.main:app --reload
pytest -q                     # 113 tests, no network required
```

---

## Deploy to Vercel

```bash
vercel            # from the repo root; or import the repo at vercel.com/new
```

No environment variables required — the default engine needs no API key.

Three things were changed to make serverless work, and they are worth knowing if
you fork this:

1. **`/api/generate` is stateless.** The browser posts the analysis back rather
   than referencing a server-side id. A serverless second request can land on a
   cold instance with neither the first request's memory nor a writable disk, so
   an id-based lookup would 404 intermittently.
2. **`/api/render/{resume|cover-letter}.{pdf|docx|txt}`** takes the document in
   the request body and returns the file, so downloads need no stored version.
   The `GET /api/export/{version_id}.{fmt}` route still exists for local use.
3. **Storage fails soft.** On a read-only filesystem `backend/db.py` marks itself
   unavailable and every call returns an empty default. Analysis, generation,
   scoring, validation and export are pure functions of their inputs and keep
   working; only version history and the application tracker go quiet.

`requirements.txt` is deliberately slim for the serverless bundle. `pdfplumber`
(better local PDF column handling) and `anthropic` add ~39 MB and are both lazily
imported — install them locally with `pip install -r requirements-dev.txt`.

**What you lose on Vercel:** version history and the application tracker, because
they need a writable disk. Everything that produces a document works. For the
tracker, run it locally or point `DB_PATH` at a hosted Postgres/Turso.

---

## Architecture

### What the rules engine does

| Stage | How |
|---|---|
| Parse resume | Section-synonym detection; roles anchored on **date ranges**, with the header read from before *or* after the date line (real resumes do both); wrapped bullets rejoined by indentation |
| Total experience | Employment intervals **merged**, so concurrent roles aren't double-counted |
| Analyse JD | Requirements split per skill; priority **P0–P3** from the posting's own section headings and hedging words ("required" vs "preferred" vs "nice to have") |
| Match | Canonical ontology + weighted semantic edges (below) |
| Position | Target title re-levelled *down* to what your titles evidence; never up |
| Write | Bullets ranked by JD relevance, selected within a word budget, kept **verbatim**; skills regrouped by requirement priority; summary template-filled from parsed fields |
| Validate | 18 deterministic ATS checks + the truthfulness gate |

Presentation-only transforms are allowed because they cannot change a fact:
stripping filler openers ("Responsible for…"), normalising dates to one format,
splitting an over-long bullet at a clause boundary, removing decorative glyphs.

### A graph, not a chain

The 12 stages form a DAG. Resume parsing and JD analysis are independent, so the
graph runs them **concurrently** — both are IO-bound LLM calls, so this is a
straight wall-clock win. `/generate` re-runs the same graph with the analysis
context already populated, so those nodes report `cached` and cost nothing.

```
┌─ parse resume ─┐                                       (parallel)
│                ├─ evidence index ─┐
├─ analyse JD ───┘                  ├─ matrix ─ refine ─┬─ gaps
└────────────────────────────────────┘                  └─ positioning
                                                             │
                          ┌──────────────────────────────────┘
                          ▼
   ╭──────────── repair loop (≤ N iterations) ────────────╮
   │  write ──▶ ATS checks + truth gate ──▶ pass? ─┐      │
   │    ▲                                          │      │
   │    └───── feedback: the exact offenders ◀─────┘      │
   ╰──────────────────────────────────────────────────────╯
                          │
        lift loop (until dry) ─▶ claim audit ─▶ recruiter ─▶ score
```

`backend/graph.py` (~200 lines, zero dependencies) gives per-level parallelism,
node caching, retries, optional nodes that degrade instead of aborting, and a
trace the UI renders as a live pipeline view.

### Loops, because one-shot generation can't guarantee correctness

`backend/services/loops.py` implements two:

**RepairLoop** — generate → validate → repair. On a critical failure it feeds the
*exact offending strings* back to the writer and regenerates. It stops on one of
three named conditions and reports which fired:

| Stop reason | Meaning |
|---|---|
| `all_critical_checks_passed` | Converged. Marked `optimized`. |
| `score_plateau` | Two iterations with no meaningful gain — stop burning tokens. |
| `iteration_cap` | Hit the configured limit. Marked `needs_review`. |

It returns the **best** candidate seen, not the last one — a later iteration that
scores worse doesn't overwrite a better earlier draft.

**lift_loop** — loop-until-dry. Finds requirements you genuinely meet but that
never appear in the document, and requests one targeted revision per round. It
stops the moment a round surfaces nothing new, so it converges rather than
running a fixed number of expensive rounds. A lift revision is discarded if it
regresses the truth gate — keyword coverage never buys a correctness regression.

### Why the matching is a curated ontology, not embeddings

`backend/services/ontology.py` holds ~163 canonical skills with alias groups and
**weighted semantic edges**. A JD asking for "container orchestration" is
satisfied at 0.92 by a resume saying "OpenShift" — via concept expansion to
`kubernetes`, then one edge hop to `openshift`.

Embeddings would also work, but they're non-deterministic, add a model
dependency, and — decisively — **can't explain themselves**. Here every match
prints its tier and its path, which is what lets the matrix be something you'd
defend in an interview.

| Tier | Score | Meaning |
|---|---|---|
| `EXACT` | 1.00 | Literal skill match |
| `STRONG_SEMANTIC` | 0.85–0.99 | Different words, same capability |
| `PARTIAL` | 0.60–0.84 | Adjacent or narrower |
| `WEAK` | 0.35–0.59 | Tangential |
| `NONE` | 0.00 | No supporting evidence |

### Scoring is code, not an opinion

```python
keyword_coverage      25%   # literal JD keyword presence (models a naive ATS)
semantic_alignment    20%   # priority-weighted mean of match scores
required_skills       15%   # P0 coverage only — optional reqs never drag this down
experience_alignment  15%   # years, domain, leadership
title_alignment        5%
evidence_strength      5%   # quantified / outcome-shaped bullets
ats_format            10%   # deterministic format checks
recruiter_readability  5%
```

Weights are configurable per request and renormalise automatically. Every
component returns its raw score, its weight, an explanation, and supporting
detail — the UI renders all of it. Same inputs always produce the same score.

Requirements are classified **P0 (mandatory) → P3 (nice-to-have)**, and missing
optional requirements deliberately don't tank the score.

---

## Project layout

```
resume-fit-ai/
├── backend/
│   ├── main.py                 FastAPI routes; keys never reach the browser
│   ├── graph.py                DAG engine — parallel, cached, traced
│   ├── config.py               env-driven settings
│   ├── db.py                   SQLite: analyses, versions, tracker
│   ├── models/schemas.py       typed contract for every stage
│   ├── llm/
│   │   ├── base.py             provider interface + JSON recovery + retry
│   │   ├── anthropic_provider.py   native output_config.format schema enforcement
│   │   ├── openai_provider.py      JSON mode + schema-in-prompt
│   │   ├── gemini_provider.py      JSON MIME + schema-in-prompt
│   │   └── mock_provider.py        offline; no network, no key
│   ├── prompts/
│   │   ├── __init__.py         system prompts per stage
│   │   └── schemas.py          JSON Schemas (Anthropic-constraint compliant)
│   └── services/
│       ├── local_engine.py     THE ENGINE — parse, analyse, position, write
│       ├── dates.py            date parsing, normalisation, interval merging
│       ├── ontology.py         canonical skills + weighted semantic edges
│       ├── matching.py         evidence index + requirement matching
│       ├── scoring.py          explainable weighted scoring
│       ├── loops.py            RepairLoop + lift_loop
│       ├── ats_validator.py    16 deterministic format/readability checks
│       ├── truth_validator.py  the hard gate
│       ├── pipeline.py         graph wiring + orchestration
│       ├── render.py           canonical plain-text rendering
│       ├── exporters.py        real DOCX / real PDF / TXT
│       ├── diffing.py          before/after classification
│       └── text_extract.py     PDF / DOCX / TXT / MD / RTF intake
├── frontend/                   index.html + styles.css + app.js (no build step)
├── tests/                      80 tests, no network required
└── samples/                    a worked resume + JD to try it on
```

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | provider, model, readiness |
| `GET` | `/api/config` | default weights + labels |
| `POST` | `/api/extract` | file → text (preview before analysing) |
| `POST` | `/api/analyze` | resume + JD → profile, matrix, gaps, positioning, baseline score |
| `POST` | `/api/generate` | analysis → tailored resume, scores, validation, diff |
| `GET` | `/api/versions` | every generated version |
| `GET` | `/api/export/{id}.{txt\|docx\|pdf}` | download |
| `GET/POST/DELETE` | `/api/applications` | application tracker |
| `GET` | `/api/analytics/positioning` | which positioning is performing best |
| `DELETE` | `/api/data` | delete everything stored locally |

### Worked example

```bash
curl -s -X POST localhost:8000/api/analyze \
  --form-string "resume_text=$(cat samples/master_resume.txt)" \
  --form-string "jd_text=$(cat samples/job_description.txt)" \
  -F "target_market=USA" > analysis.json

ID=$(jq -r .analysis_id analysis.json)

curl -s -X POST localhost:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d "{\"analysis_id\":\"$ID\",\"max_repair_iterations\":3,\"lift_rounds\":1}" > result.json

VER=$(jq -r .version_id result.json)
curl -sO -J localhost:8000/api/export/$VER.pdf
```

The bundled sample pairs a genuine Senior Backend Engineer resume against a Staff
Platform JD, so it exercises the interesting paths: OpenShift matching Kubernetes
semantically, Python and Rust reported as honest gaps, and a Staff-vs-Senior
level mismatch that the positioning engine flags rather than papering over.

---

## The application tracker and the learning loop

Every generated version is saved with its scores. Log outcomes against them and
`/api/analytics/positioning` aggregates interview and offer rates **by
positioning** — so you can see whether "Senior Java Backend" or "AI Platform
Engineer" is actually getting you interviews.

It flags any positioning with fewer than 8 applications as `low sample`. A 100%
interview rate from two applications is noise, and the tool says so rather than
letting you over-fit to it.

The learning loop only ever changes *emphasis*. It never changes what's true.

---

## Providers

Set `LLM_PROVIDER` in `.env`.

| Provider | Structured output | Notes |
|---|---|---|
| `local` *(default)* | n/a — no model | Pure rules; see above |
| `ollama` | n/a | Local model rewords bullets only; a rewrite that adds a number or drops a technology is rejected and the original restored |
| `anthropic` | Native, server-enforced | `output_config.format` with a JSON Schema |
| `openai` | JSON mode + schema in prompt | `pip install openai` |
| `gemini` | JSON MIME + schema in prompt | `pip install google-genai` |
| `mock` | Deterministic stubs | No network, no key |

The hosted paths are optional. The Anthropic one is written against the current
API, and matters only if you switch to it:

- **`temperature` / `top_p` / `top_k` are rejected with a 400** on Claude Opus 5,
  Fable 5, Opus 4.8 and 4.7. Never send them.
- Structured JSON is `output_config.format`, not the deprecated `output_format`
  and not assistant prefill (prefills 400 on every current model).
- Thinking is on by default on Opus 5, and `max_tokens` caps thinking **plus**
  response text — hence the generous per-stage budgets.
- `stop_reason` is checked before reading `content`: a refusal returns HTTP 200
  with empty content, so naive `content[0]` indexing would crash.
- The master resume + JD ride in a **cached** system block, so later stages read
  the shared prefix at ~0.1× cost.

A typical full run is ~7 model calls (parse, JD, refine, positioning, write,
claim audit, recruiter), plus one extra write per repair or lift iteration.

---

## Privacy

Your resume is sensitive, and this app treats it that way.

- Runs entirely on your machine. Data lives in `data/resumefit.db`.
- API keys are read by the backend only. **The browser never receives a key** —
  the flow is always `frontend → backend → provider`.
- `.env` and `data/` are gitignored.
- `DELETE /api/data` removes everything, and the UI has a button for it.
- `LLM_TRACE=1` writes full prompts to `data/llm_trace.jsonl` for debugging.
  That file will contain your resume. It's off by default; delete it when done.

---

## Tests

```bash
pytest -q                            # 113 tests, no network, no API key
pytest tests/test_validators.py -q   # the adversarial truthfulness suite
```

Coverage is concentrated where the risk is: the truthfulness gate (11 adversarial
fabrication tests), the matching engine, the scoring engine's determinism and
bounds, graph semantics (cycles, caching, retries, optional-node degradation),
loop convergence (all three stop reasons, best-candidate selection, dry
detection), and export integrity — the DOCX test asserts the OOXML contains
**no `<w:tbl>`, no `<w:drawing>`, no text boxes**, and the PDF test extracts the
text layer back out to prove an ATS could read it.

---

## What this will never do

It will not invent skills, achievements, metrics, years, titles, employers,
dates, certifications, education, clearances or visa status. It will not stuff
keywords, hide a missing mandatory requirement, guarantee an interview, or
promise an ATS score.

If the JD needs something you can't support, it tells you, explains the risk, and
recommends **against** adding it.

Maximum relevance from real experience. That's the whole product.

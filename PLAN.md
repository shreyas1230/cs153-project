# Implementation Plan: Cross-Domain Scientific Claim Conflict Detector

## Context

Shreyas Agarwal (shrey15@stanford.edu) is a Stanford CS 153 student building a research tool for his course final project. The project is titled **"Detecting and Aligning Inconsistent Claims Across Scientific Papers with Language Models"** and is due **June 4, 2026** (9 days from today). Nothing is built yet.

The core insight of the project is the opposite of existing LLM tools: instead of synthesizing papers into confident summaries that hide disagreements, this tool makes disagreement and definitional drift **first-class outputs**. A researcher uploads PDFs and asks a question; the system surfaces which papers support vs. contradict each other, and—critically—whether apparent disagreements are real or merely due to each field using the same word differently.

Rubric weighting that should drive all scope decisions: Problem & Insight (3 pts), **Execution & Technical Work (5 pts)**, Evaluation & Evidence (3 pts), Communication & Presentation (2 pts), Process & Disclosure (2 pts).

Stack chosen: **Python/FastAPI backend + Next.js/React frontend**. LLM inference via **Cloudflare Workers AI** (covered by $100K Cloudflare credits). Backend hosted on **DigitalOcean** ($250 credits). Frontend on **Cloudflare Pages** (free from Cloudflare credits). Total additional out-of-pocket cost: **$0**.

---

## GitHub Note

The GitHub account for this project (`shrey15@stanford.edu`) is **different from the one configured on this WSL machine**. Before the first commit, configure git locally in the project directory:

```bash
git config user.name "Shreyas Agarwal"
git config user.email "shrey15@stanford.edu"
```

Do NOT use `--global` — this keeps the WSL default account untouched. Also set up a separate SSH key or use HTTPS with a Personal Access Token for the Stanford GitHub account when pushing.

---

## Architecture

### High-Level Flow

```
User (browser)
    │
    ▼
Next.js Frontend (Cloudflare Pages)
    │  multipart: PDFs + question
    ▼
FastAPI Backend (DigitalOcean Droplet)
    │
    ├─ 1. Ingest: parse PDFs with pdfplumber → text per page
    ├─ 2. Claim Extraction (LLM call per paper)      ┐
    ├─ 3. Terminology Normalization (LLM call)        ├─ results cached
    ├─ 4. Conflict Detection (LLM call)              ┘  as sessions/{id}.json
    └─ 5. Return structured AnalysisResult
```

### Compute Resources (Zero Additional Cost)

| Resource | Provider | Used for |
|---|---|---|
| DigitalOcean Droplet ($250 credits) | DigitalOcean | FastAPI backend server |
| Workers AI ($50K cap from $100K Cloudflare credits) | Cloudflare | LLM inference — primary (free) |
| Cloudflare Pages | Cloudflare | Next.js frontend hosting |
| Cloudflare R2 | Cloudflare | Temporary PDF storage |
| Cloudflare AI Gateway | Cloudflare | Unified API endpoint + cost observability |

### LLM Strategy: Start Free, Upgrade If Needed

**Phase 1 (default):** `@cf/meta/llama-3.3-70b-instruct` via Cloudflare Workers AI — zero cost, test quality on real papers on Day 1.

**Phase 2 (if quality insufficient):** Switch to `claude-sonnet-4-6` via Anthropic API — ~$10–20 total for the project. Sonnet is the right tier: better than Opus on cost/quality ratio for structured extraction. Opus 4.7 is roughly 5× more expensive per token than Sonnet for no meaningful gain on claim extraction or JSON formatting tasks.

**Switching is a 2-line config change** because all LLM calls go through a single `pipeline/llm_client.py` abstraction:

```python
# pipeline/llm_client.py
import os, httpx
from anthropic import Anthropic

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "cloudflare")  # or "anthropic"

async def chat(messages: list[dict], json_mode=True) -> str:
    if LLM_PROVIDER == "anthropic":
        client = Anthropic()
        resp = client.messages.create(model="claude-sonnet-4-6", messages=messages, ...)
        return resp.content[0].text
    else:
        # Cloudflare Workers AI
        resp = httpx.post(CF_WORKERS_AI_URL, headers=..., json={"messages": messages, ...})
        return resp.json()["result"]["response"]
```

Changing `LLM_PROVIDER=anthropic` in `.env` is the entire switch. No other code changes.

### PDF Handling

Cloudflare Workers AI does not natively accept PDF documents. PDFs are parsed to text using **pdfplumber** in the FastAPI backend before being sent to the LLM. For scientific papers, pdfplumber handles most layouts well. Scanned PDFs (image-only) are flagged to the user with a warning. If the switch to Claude Sonnet is made, pdfplumber remains in the stack — it's lightweight and its output is valid text for any LLM.

---

## Directory Structure

```
project/
├── backend/
│   ├── main.py                  # FastAPI app, /api/analyze, /api/status, /api/results
│   ├── pipeline/
│   │   ├── models.py            # Pydantic schemas (Claim, ClaimPair, TermConflict, AnalysisResult)
│   │   ├── llm_client.py        # Provider abstraction (Cloudflare Workers AI / Anthropic)
│   │   ├── extract.py           # Stage 1: claim extraction per paper
│   │   ├── normalize.py         # Stage 2: cross-paper terminology normalization
│   │   └── detect.py            # Stage 3: conflict detection and classification
│   ├── eval/
│   │   ├── harness.py           # CLI: precision/recall vs. hand-labeled ground truth
│   │   └── ground_truth/        # JSONL files with known conflicts per paper set
│   ├── sessions/                # Runtime JSON result storage (gitignored)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Upload + question input
│   │   └── results/[id]/page.tsx # Claim map view
│   ├── components/
│   │   ├── PDFUpload.tsx        # Drag-and-drop zone
│   │   ├── ClaimMap.tsx         # Main results layout
│   │   ├── ConflictCard.tsx     # Single claim-pair card (color coded)
│   │   └── SourceDrawer.tsx     # Slide-in panel showing original passage + paper
│   └── package.json
└── README.md                    # AI disclosure + setup instructions (required by rubric)
```

---

## Data Schemas (backend/pipeline/models.py)

```python
class Claim(BaseModel):
    id: str
    paper_title: str
    paper_index: int          # index into uploaded PDF list
    claim_text: str
    source_passage: str       # verbatim excerpt from paper
    page_number: Optional[int]
    key_terms: list[str]
    domain_signals: list[str] # e.g. ["ML", "neuroscience"]

class TermDefinition(BaseModel):
    term: str
    paper_index: int
    definition: str           # how this paper uses the term

class TermConflict(BaseModel):
    term: str
    definitions: list[TermDefinition]
    conflict_type: Literal["CONSISTENT", "RELATED", "DIVERGENT"]
    explanation: str

class ClaimPair(BaseModel):
    claim_a_id: str
    claim_b_id: str
    relationship: Literal["SUPPORT", "CONTRADICT", "INCOMMENSURABLE"]
    explanation: str
    terminology_note: Optional[str]  # populated for INCOMMENSURABLE

class AnalysisResult(BaseModel):
    session_id: str
    question: str
    papers: list[str]              # paper titles
    claims: list[Claim]
    term_conflicts: list[TermConflict]
    claim_pairs: list[ClaimPair]
```

---

## Three-Stage Prompting Strategy

### Stage 1 — Claim Extraction (extract.py)

One LLM call per paper. Text extracted via pdfplumber first.

**System prompt:**
> You extract atomic, falsifiable claims from scientific papers. Each claim must be: a single specific assertion; grounded in evidence from the paper; accompanied by the verbatim passage it comes from; tagged with key terms the claim depends on and the paper's apparent research domain. Do not paraphrase — copy the source passage exactly. Return a JSON object matching this schema: { "claims": [...] }

JSON schema embedded in system prompt (Workers AI does not support function/tool calling like the Anthropic API, so we use JSON mode + schema-in-prompt).

### Stage 2 — Terminology Normalization (normalize.py)

One LLM call with all claims across papers.

**System prompt:**
> You are given claims extracted from multiple scientific papers, each tagged with key terms. Identify all cases where papers use the same term to mean different things (DIVERGENT), different terms for the same concept (RELATED), or consistent usage (CONSISTENT). Return a JSON object: { "glossary": [...] }

### Stage 3 — Conflict Detection (detect.py)

One LLM call. Pass all claims + the glossary from Stage 2.

**System prompt:**
> Given these claims from multiple papers and their terminology glossary, identify pairs of claims that: (1) SUPPORT each other; (2) CONTRADICT each other; (3) INCOMMENSURABLE — appear to disagree but are answering different questions, using different methods, or suffering from terminology drift. For INCOMMENSURABLE pairs, explain exactly why they cannot be directly compared. Return JSON: { "pairs": [...] }

**Why three separate calls over one monolithic prompt:** Each stage has a different failure mode. Claim extraction hallucinates when also asked to judge conflicts. Conflict detection is more accurate when given a pre-built glossary. Separating stages also makes evaluation and debugging possible per-stage.

---

## Backend API (main.py)

| Endpoint | Method | Description |
|---|---|---|
| `/api/analyze` | POST | Multipart: PDFs + `question` string. Returns `{session_id}` immediately; processing runs async via BackgroundTasks. |
| `/api/status/{session_id}` | GET | Returns `{stage: "extracting"\|"normalizing"\|"detecting"\|"done", progress: 0-100}` |
| `/api/results/{session_id}` | GET | Returns full `AnalysisResult` JSON |

Sessions stored as `sessions/{session_id}.json`. No database needed at this scope.

---

## Frontend UI

### Page 1 — Upload (`app/page.tsx`)

- Drag-and-drop PDF zone (2–10 papers)
- Text input: "What question are you trying to answer across these papers?"
- "Analyze" button → POST to `/api/analyze` → poll `/api/status` with progress bar → navigate to results

### Page 2 — Claim Map (`app/results/[id]/page.tsx`)

Split-panel layout:

**Left panel (45%):**
- Header: summary stats (N papers, N claims, N conflicts, N terminology divergences)
- Terminology Divergences section: cards per divergent term showing per-paper definitions
- Claim Conflicts section: cards per claim pair, color-coded:
  - Green border: SUPPORT
  - Red border: CONTRADICT
  - Yellow border: INCOMMENSURABLE
- Each card shows: claim A (truncated) vs. claim B (truncated) + relationship label + one-line explanation

**Right panel (55%):**
- Activated when user clicks any claim card
- Shows paper title + source passage (verbatim, highlighted)
- Shows key terms highlighted inline
- If INCOMMENSURABLE: shows terminology note explaining the drift

This layout is achievable in 2 days with Tailwind CSS and no graph library. It is more readable for a 3-minute demo than a force-directed graph and takes half the time to build.

---

## Evaluation Harness (eval/harness.py)

**Ground truth format (JSONL):**
```json
{"paper_set": ["p1.pdf", "p2.pdf"], "question": "...", "conflicts": [{"claim_a_text": "...", "claim_b_text": "...", "type": "CONTRADICT"}]}
```

**Metrics reported:**
- Precision = correctly flagged conflicts / total flagged
- Recall = correctly flagged conflicts / total ground truth conflicts
- F1, broken down by relationship type (SUPPORT / CONTRADICT / INCOMMENSURABLE)

**Target:** ~15 papers spanning 2 cross-domain pairs (e.g., ML + cognitive science, climate science + economics). Hand-label 5 papers first (Day 1-3 validation), expand to 15 by Day 7.

This is the evidence section of the rubric (3 pts) — must report actual numbers in the README and demo video.

---

## 9-Day Implementation Schedule

| Day | Date | Work |
|---|---|---|
| 1 | May 26 | Git repo + FastAPI skeleton + Next.js scaffold. Set up Cloudflare AI Gateway + Workers AI. Smoke test: parse one PDF with pdfplumber, send text to Workers AI, get JSON back. |
| 2 | May 27 | `extract.py`: claim extraction, test on 2–3 hand-read papers, iterate prompt until claims match expectations. |
| 3 | May 28 | `normalize.py` + `detect.py`: wire all three stages. End-to-end CLI run on 5 papers. |
| 4 | May 29 | Evaluation harness + hand-label ground truth for 5-paper set. Measure baseline precision/recall. Fix biggest prompt failure mode. |
| 5 | May 30 | Frontend page 1: PDF upload + question + POST to backend + polling progress bar. |
| 6 | May 31 | Frontend page 2: claim map split panel, ConflictCard, SourceDrawer, color coding. |
| 7 | June 1 | Use tool on real literature review for a full day. Log every miss and false positive. Fix top 3. Expand eval to 15 papers. Final eval run → record numbers. |
| 8 | June 2 | README: setup instructions, AI disclosure, eval numbers. Code cleanup. Deploy backend to DigitalOcean. Deploy frontend to Cloudflare Pages. |
| 9 | June 3–4 | Record 3-minute demo video (Q1–Q4 from rubric). Verify live deployment. GitHub submission. |

---

## Key Technical Decisions

1. **Workers AI JSON mode + schema-in-prompt** — set `response_format: {type: "json_object"}` and embed the target schema in the system prompt. Belt-and-suspenders for reliable structured output without tool-calling support.

2. **Cloudflare AI Gateway as single LLM endpoint** — all calls go through `https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/workers-ai/`. Cost dashboard and request logs come free.

3. **temperature=0 throughout** — factual extraction; determinism makes evaluation reproducible.

4. **No embeddings / vector DB** — LLM reasons over full claim set. Llama 3.3 70B has 128K context, sufficient for ≤20 papers. Embeddings add complexity with no quality benefit at this scale.

5. **Session JSON files, not a database** — sufficient, reproducible, inspectable. `sessions/` is gitignored.

6. **DigitalOcean Droplet + Cloudflare Pages** — both covered by provided credits. Both produce stable public URLs needed for the demo video.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Workers AI returns malformed JSON | JSON mode + schema in prompt; retry 3× with backoff; fallback: regex extract last JSON block |
| Very long papers overflow context | Chunk by section heading; extract per chunk and merge; most papers fit whole in 128K |
| High false-positive rate in conflict detection | temp=0; optional second "verification pass" for borderline CONTRADICTs |
| Workers AI quality insufficient for scientific reasoning | Switch to `claude-sonnet-4-6` via Anthropic API (2-line env change); disclose in README |
| 9 days too short | Split-panel table UI ships before any graph. Pipeline quality + eval numbers > UI polish for grade. |
| pdfplumber struggles with a specific paper | Fallback: accept plain-text paste as input instead of PDF |

---

## Rubric Alignment Checklist

- **Problem & Insight (3 pts)**: The "anti-synthesis" framing is original and compelling — proposal is well-written.
- **Execution (5 pts)**: Working web app + functional 3-stage pipeline = full credit zone. Demo must show end-to-end flow with real papers.
- **Evaluation (3 pts)**: Precision/recall/F1 numbers from hand-labeled eval set. Report in README and mention in video.
- **Communication (2 pts)**: Clear README with `pip install` + `npm install` setup, demo video walks through a real use case.
- **Integrity (2 pts)**: Document Claude Code and AI tool usage in README. Public repo with meaningful commit history from Day 1 onward.

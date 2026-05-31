# Detecting and Aligning Inconsistent Claims Across Scientific Papers with Language Models

**CS 153: Frontier Systems | Stanford Spring 2026 | Shreyas Agarwal (shrey15@stanford.edu)**

A cross-domain literature reading assistant that surfaces conflicts and terminology mismatches between papers rather than synthesizing them away. Upload PDFs, ask a question — the system shows which papers support each other, which genuinely contradict, and which only *appear* to conflict because each field uses the same word differently.

---

## Demo

**Live app:** https://cs153-project-6on.pages.dev/ (frontend on Cloudflare Pages, backend API on DigitalOcean)

**Demo video:** _(3-minute walkthrough — link added on submission)_

> Upload "Attention Is All You Need" + an Attention Schema neuroscience paper, ask *"What is attention and how does it work?"* → The system correctly identifies that both papers use the word "attention" to mean entirely different things, and explains why they cannot be directly compared.

---

## How It Works

Three-stage LLM pipeline:

1. **Claim Extraction** — one LLM call per paper (parallel); extracts atomic, falsifiable claims tagged with verbatim source passages, key terms, and domain signals
2. **Terminology Normalization** — one LLM call across all claims; builds a cross-paper glossary flagging DIVERGENT, RELATED, or CONSISTENT term usage
3. **Conflict Detection** — one LLM call using claims + glossary; classifies cross-paper claim pairs as SUPPORT, CONTRADICT, or INCOMMENSURABLE

Results are displayed in an interactive split-panel claim map. Clicking any conflict card shows the verbatim source passage from the original PDF.

By default the analysis is **question-blind** — extraction and normalization see only the papers, so the system surfaces *all* cross-paper conflicts and terminology drift, including ones you didn't think to ask about. An optional **"Focus on my question"** toggle steers only the detection stage toward question-relevant relationships; it is off by default (the design rationale and the supporting ablation are in [REPORT.md §4.5](REPORT.md)).

Full technical details in [REPORT.md](REPORT.md).

---

## Evaluation Results

Evaluated on a **13-pair benchmark across 10 papers**, exercising all three relationship types. Ground truth for the CONTRADICT and SUPPORT categories is drawn from *documented relationships in the literature* (e.g. Santurkar et al. challenging the internal-covariate-shift explanation of batch normalization), so those labels do not depend on the model's own output. All numbers are **mean ± std over 5 independent runs**:

Two configurations are compared: a *monolithic baseline* that does extraction, normalization, and detection in a single LLM prompt (a reference point only), and the *three-stage pipeline* — separate extract → normalize → detect stages — which is what the app actually runs out of the box ("shipped default").

| Configuration (13 pairs) | Precision | Recall | F1 |
|---|---|---|---|
| Monolithic single-prompt baseline | 0.17 ± 0.02 | 0.62 ± 0.04 | 0.24 ± 0.02 |
| **Three-stage pipeline (shipped default)** | **0.30 ± 0.03** | **0.62 ± 0.07** | **0.35 ± 0.04** |

The three-stage pipeline beats the monolithic baseline by +0.11 F1, entirely on precision at equal recall. Known **CONTRADICT** pairs are labeled correctly in 9/10 runs and **SUPPORT** in 5/10. Reported precision is a lower bound (ground truth lists only the canonical relationship per pair, so additional valid pairs count as false positives).

> An earlier evaluation reported F1 = 0.68 on just 3 pairs, but it used model-derived ground truth (the model graded against its own output) and a single run. The lower numbers here reflect a larger, non-circular, variance-aware benchmark — a harder and more honest evaluation, not a regression. See [REPORT.md §4](REPORT.md) for the full results, the question-steering ablation, and the confusion matrix.

---

## Setup

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENROUTER_API_KEY
uvicorn main:app --port 8001
```

### Frontend

```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8001" > .env.local
npm run dev -- --port 3003
```

Open http://localhost:3003. Upload 2+ PDF papers, type a research question, click Analyze.

### Run Evaluation

```bash
cd backend && source .venv/bin/activate

# Full benchmark, 5 runs with mean +/- std (runs pipeline, ~$1-2 total)
python -m eval.harness --ground-truth eval/ground_truth/all_pairs.jsonl \
  --papers-dir test_papers --preds-file eval/preds_qaware_5runs.json --save-preds --runs 5

# Re-score from saved predictions (free, reproducible)
python -m eval.harness --ground-truth eval/ground_truth/all_pairs.jsonl \
  --papers-dir test_papers --preds-file eval/preds_qaware_5runs.json

# Monolithic single-prompt baseline, same scoring harness
python -m eval.harness --ground-truth eval/ground_truth/all_pairs.jsonl \
  --papers-dir test_papers --preds-file eval/preds_baseline_3runs.json --save-preds --runs 3 --baseline
```

Test papers — download from arXiv into `backend/test_papers/` (`wget https://arxiv.org/pdf/<id>.pdf`):
`1706.03762`, `2402.01056`, `1612.00796`, `1502.03167`, `1805.11604`, `1412.6980`, `1705.08292`, `1512.03385`, `1505.00387`, `1810.04805`.

---

## Stack

- **Backend:** Python 3.12, FastAPI, pdfplumber, httpx, Pydantic
- **Frontend:** Next.js 16, Tailwind CSS
- **LLM:** `meta-llama/llama-3.3-70b-instruct` via OpenRouter, pinned to the Nebius provider for run-to-run consistency (~$0.003/analysis)
- **Original LLM (free tier):** Cloudflare Workers AI (hit 10k neuron/day limit)
- **Deployment:** Frontend as a static export on Cloudflare Pages (https://cs153-project-6on.pages.dev/); FastAPI backend on DigitalOcean (https://claims-lens-i256v.ondigitalocean.app)

---

## AI Tool Disclosure

This project was built with [Claude Code](https://claude.ai/code) (claude-sonnet-4-6) as an AI pair programmer throughout development. Specifically, Claude Code was used for:

- **Scaffolding** — initial FastAPI and Next.js project structure
- **LLM prompts** — drafting and iterating on the system prompts for all three pipeline stages
- **Debugging** — diagnosing the OpenRouter `response_format` / tool-call misrouting bug, fixing the within-paper pair contamination issue
- **Evaluation harness** — writing `eval/harness.py`, `eval/diagnose.py`, and the ground truth JSONL files
- **Robustness pass** — expanding the benchmark to 13 pairs with literature-grounded CONTRADICT/SUPPORT labels, adding variance/baseline/confusion-matrix support to the harness, wiring the previously-unused `question` parameter into the detection stage as an opt-in, and running the evaluation sweeps
- **Report** — writing `REPORT.md`

All code was reviewed and understood by the author. The architecture decisions, the three-way relationship taxonomy (SUPPORT / CONTRADICT / INCOMMENSURABLE), the anti-synthesis framing, and the evaluation methodology are the author's own. No base repositories were forked; all code is original.

Per CS 153 AI policy: AI tool usage is disclosed here in the README as required.

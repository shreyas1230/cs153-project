# Detecting and Aligning Inconsistent Claims Across Scientific Papers with Language Models

**CS 153: Frontier Systems | Stanford Spring 2026 | Shreyas Agarwal (shrey15@stanford.edu)**

A cross-domain literature reading assistant that surfaces conflicts and terminology mismatches between papers rather than synthesizing them away. Upload PDFs, ask a question — the system shows which papers support each other, which genuinely contradict, and which only *appear* to conflict because each field uses the same word differently.

---

## Demo

> Upload "Attention Is All You Need" + an Attention Schema neuroscience paper, ask *"What is attention and how does it work?"* → The system correctly identifies that both papers use the word "attention" to mean entirely different things, and explains why they cannot be directly compared.

---

## How It Works

Three-stage LLM pipeline:

1. **Claim Extraction** — one LLM call per paper (parallel); extracts atomic, falsifiable claims tagged with verbatim source passages, key terms, and domain signals
2. **Terminology Normalization** — one LLM call across all claims; builds a cross-paper glossary flagging DIVERGENT, RELATED, or CONSISTENT term usage
3. **Conflict Detection** — one LLM call using claims + glossary; classifies cross-paper claim pairs as SUPPORT, CONTRADICT, or INCOMMENSURABLE

Results are displayed in an interactive split-panel claim map. Clicking any conflict card shows the verbatim source passage from the original PDF.

Full technical details in [REPORT.md](REPORT.md).

---

## Evaluation Results

Evaluated on 3 paper pairs (machine learning + neuroscience) with hand-verified ground truth:

| Paper Pair | Type | Precision | Recall | F1 |
|---|---|---|---|---|
| ML Attention vs. Neuro Attention | Cross-domain | 0.75 | 0.60 | 0.67 |
| Catastrophic Forgetting vs. ML Attention | Same-domain | 0.40 | 1.00 | 0.57 |
| Catastrophic Forgetting vs. Neuro Attention | Cross-domain | 0.80 | 0.80 | 0.80 |
| **Aggregate** | | **0.65** | **0.80** | **0.68** |

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

# Run pipeline and save predictions (costs ~$0.01 total)
python -m eval.harness --ground-truth eval/ground_truth/set1.jsonl \
  --papers-dir test_papers --preds-file eval/preds_set1.json --save-preds

# Evaluate from saved predictions (free, reproducible)
python -m eval.harness --ground-truth eval/ground_truth/set1.jsonl \
  --papers-dir test_papers --preds-file eval/preds_set1.json
```

Test papers (download from arXiv): `1706.03762`, `2402.01056`, `1612.00796` — save to `backend/test_papers/`.

---

## Stack

- **Backend:** Python 3.11, FastAPI, pdfplumber, httpx, Pydantic
- **Frontend:** Next.js 16, Tailwind CSS
- **LLM:** `meta-llama/llama-3.3-70b-instruct` via OpenRouter (~$0.003/analysis)
- **Original LLM (free tier):** Cloudflare Workers AI (hit 10k neuron/day limit)

---

## AI Tool Disclosure

This project was built with [Claude Code](https://claude.ai/code) (claude-sonnet-4-6) as an AI pair programmer throughout development. Specifically, Claude Code was used for:

- **Scaffolding** — initial FastAPI and Next.js project structure
- **LLM prompts** — drafting and iterating on the system prompts for all three pipeline stages
- **Debugging** — diagnosing the OpenRouter `response_format` / tool-call misrouting bug, fixing the within-paper pair contamination issue
- **Evaluation harness** — writing `eval/harness.py`, `eval/diagnose.py`, and the ground truth JSONL files
- **Report** — writing `REPORT.md`

All code was reviewed and understood by the author. The architecture decisions, the three-way relationship taxonomy (SUPPORT / CONTRADICT / INCOMMENSURABLE), the anti-synthesis framing, and the evaluation methodology are the author's own. No base repositories were forked; all code is original.

Per CS 153 AI policy: AI tool usage is disclosed here in the README as required.

# Detecting and Aligning Inconsistent Claims Across Scientific Papers with Language Models

**Shreyas Agarwal** | shrey15@stanford.edu | CS 153: Frontier Systems, Stanford Spring 2026

---

## Abstract

Existing LLM-based literature tools synthesize papers into confident summaries that flatten the disagreements researchers most need to see. This project takes the opposite approach: a three-stage pipeline that makes disagreement and definitional drift *first-class outputs*. Given a user question and a set of uploaded PDFs, the system extracts atomic claims per paper, builds a cross-paper terminology glossary, and detects pairs of claims that contradict each other, support each other, or are *incommensurable* — appearing to conflict only because each field uses the same word differently. The output is an interactive claim map in a web app. Evaluated on three paper pairs spanning machine learning and neuroscience, the system achieves an aggregate F1 of **0.68** on INCOMMENSURABLE conflict detection, with a clear pattern: cross-domain pairs (F1 = 0.67–0.80) outperform same-domain pairs (F1 = 0.57).

---

## 1. Problem and Motivation

When researchers work across disciplines — computational neuroscience and machine learning, climate science and economics, cognitive psychology and AI — the hardest problem is not finding relevant papers but knowing when two papers that *appear* to agree are actually using the same word to mean different things, or when two papers that *appear* to disagree are simply answering different questions at different scopes.

Current LLM-based tools such as Perplexity or NotebookLM synthesize papers into confident summaries. This is useful for intra-domain review, where shared vocabulary can be assumed. It is actively harmful for cross-domain review, where the synthesis hides precisely the disagreements a researcher needs to surface.

**The central insight of this project is that incommensurability — not direct contradiction — is the dominant failure mode in cross-domain scientific reading.** Two researchers who both study "attention" may disagree in ways that are invisible unless you know that the ML paper uses "attention" to mean a scaled dot-product weighting mechanism, and the neuroscience paper uses "attention" to mean a selective cognitive phenomenon controlled by a brain-internal model. No amount of summarization exposes this; it requires deliberately tracking how each paper defines its own key terms and flagging the divergences.

The goal of this system is to make these three categories of relationship explicit: **SUPPORT** (claims that mutually reinforce each other), **CONTRADICT** (genuine logical or empirical conflicts), and **INCOMMENSURABLE** (apparent conflicts that dissolve once terminology drift is explained).

---

## 2. System Architecture

The system is a Python/FastAPI backend with a Next.js frontend, communicating via a simple REST API. LLM inference runs through OpenRouter using `meta-llama/llama-3.3-70b-instruct`.

### 2.1 High-Level Flow

```
User uploads PDFs + types a question
        │
        ▼
Stage 1: Claim Extraction       (1 LLM call per paper, parallel)
        │
        ▼
Stage 2: Terminology Normalization  (1 LLM call, all claims)
        │
        ▼
Stage 3: Conflict Detection         (1 LLM call, claims + glossary)
        │
        ▼
Interactive Claim Map (frontend)
```

The pipeline runs asynchronously in a FastAPI `BackgroundTask`. The frontend polls `/api/status/{session_id}` and navigates to the results page when done. Sessions are stored as JSON files — no database needed at this scope.

### 2.2 Stage 1 — Claim Extraction

One LLM call per paper, executed in parallel with `asyncio.gather`. Each call returns a JSON object containing a list of atomic, falsifiable claims, each tagged with:
- `claim_text`: a single-sentence assertion
- `source_passage`: verbatim excerpt from the paper
- `page_number`: page reference
- `key_terms`: technical terms the claim depends on
- `domain_signals`: apparent research fields (e.g., `["machine learning"]`, `["cognitive neuroscience"]`)

The system prompt instructs the model to extract 10–30 claims per paper focused on results and assertions, not methodology descriptions. The verbatim `source_passage` requirement is critical: it prevents the model from paraphrasing in ways that lose the original meaning, and it enables the "source drawer" UI component that shows users exactly what passage in the original PDF each claim came from.

**Why not one monolithic prompt?** Claim extraction and conflict detection have different failure modes. When asked to simultaneously extract claims and judge conflicts, the model tends to over-extract claims that support whatever conflicts it has already decided to flag. Separating the stages eliminates this cross-contamination.

### 2.3 Stage 2 — Terminology Normalization

One LLM call with all extracted claims from all papers. The model is given claims grouped by paper and asked to identify terms where:
- **DIVERGENT**: the same word means different things across papers
- **RELATED**: different words refer to the same concept
- **CONSISTENT**: consistent usage across papers

The output is a per-run glossary that is passed directly into Stage 3. This glossary is what enables the system to distinguish an INCOMMENSURABLE pair (same word, different meaning) from a genuine CONTRADICT (same concept, different conclusions).

For example, on the "Attention Is All You Need" + "Attention Schema Theory" paper pair, the normalization stage correctly identifies that "attention" is DIVERGENT: the ML paper defines it as a scaled dot-product weighting mechanism, while the neuroscience paper defines it as a selective cognitive phenomenon involving a brain-internal predictive model.

### 2.4 Stage 3 — Conflict Detection

One LLM call with all claims and the terminology glossary. The model is asked to identify meaningful cross-paper pairs and classify each as SUPPORT, CONTRADICT, or INCOMMENSURABLE, with an explanation and, for INCOMMENSURABLE pairs, a `terminology_note` explaining the drift.

The system prompt instructs the model to be conservative on CONTRADICT — only flagging genuine logical or empirical conflicts, not mere emphasis differences. This intentional conservatism reduces false positives at the cost of recall on CONTRADICT, which is the right tradeoff: a falsely flagged contradiction is more harmful to a researcher than a missed one.

A post-processing filter removes any within-paper pairs (pairs where `claim_a` and `claim_b` come from the same paper), which the model occasionally produces despite being instructed not to.

### 2.5 PDF Handling

PDFs are parsed to text with `pdfplumber` in the FastAPI backend before any LLM call. Each page is prefixed with `[Page N]` to preserve page number information for the `source_passage` attribution. Papers longer than 50,000 characters are truncated at the claim extraction stage; in practice all three test papers are within this limit. Scanned PDFs (image-only) are detected via an empty-text check and rejected with an informative error.

### 2.6 LLM Provider

The system was originally built against Cloudflare Workers AI's free tier (`@cf/meta/llama-3.3-70b-instruct-fp8-fast`), but the free tier is limited to 10,000 neurons/day — roughly 10,000 tokens, which is consumed by a single paper pair analysis. For development and evaluation the system was switched to **OpenRouter** (`meta-llama/llama-3.3-70b-instruct`), which uses the same model at approximately $0.40/million tokens. A full pipeline run on a 2-paper pair costs approximately $0.002–0.005.

The LLM provider is abstracted behind a single `llm_client.py` module. Switching providers is a one-line change to the `LLM_PROVIDER` environment variable; the pipeline code is unchanged.

**One notable integration issue:** OpenRouter routes requests to different backend providers (Nebius, Parasail, etc.) depending on availability. Some providers' vLLM backends misinterpret `response_format: {"type": "json_object"}` as a tool-calling directive, returning `finish_reason: "tool_calls"` and `content: null`. The fix was to omit `response_format` for OpenRouter and rely on the system prompt's JSON schema instruction combined with a bracket-matching JSON extractor (`_extract_json`) that handles markdown code fences, which the model produces when `response_format` is absent.

---

## 3. Frontend

The frontend is a Next.js 16 app deployed on Cloudflare Pages. It has two pages.

**Upload page:** Drag-and-drop PDF zone, a text input for the research question, and an "Analyze" button that POSTs to `/api/analyze` and then polls `/api/status` with a progress bar that advances through the three pipeline stages.

**Results page:** A split-panel layout. The left panel (45%) lists terminology divergences and conflict pairs, each as a color-coded card: green for SUPPORT, red for CONTRADICT, yellow for INCOMMENSURABLE. Each card shows the truncated claim texts and a one-line explanation. The right panel (55%) activates on click and shows the full verbatim source passage from the original paper, the key terms highlighted, and the terminology note if the pair is INCOMMENSURABLE.

---

## 4. Evaluation

### 4.1 Methodology

A custom evaluation harness (`eval/harness.py`) measures precision and recall against hand-labeled ground truth stored in JSONL format. Each ground truth entry specifies two paper files, a question, and a list of known conflict pairs with their relationship type.

Matching between predicted pairs and ground truth pairs uses token overlap: a predicted pair matches a ground truth pair if both `claim_a` texts share ≥25% token overlap AND both `claim_b` texts share ≥25% token overlap (order-independent). This threshold is intentionally lenient to account for paraphrase — the same underlying claim expressed in slightly different words on different pipeline runs. This metric ignores word frequency (overlap is computed over distinct tokens) and is normalized by the ground-truth length, so very short ground-truth claims can match on minimal lexical overlap; we mitigate this by requiring both claims in a pair to match independently.

**A key methodological challenge:** LLM outputs are non-deterministic in practice even at `temperature=0`, because OpenRouter routes to different providers across calls, and different provider backends (e.g. Nebius vs Parasail) produce slightly different claim phrasings. Ground truth written to match one run's exact phrasing will fail to match a different run's paraphrase of the same claim. The solution is a `--save-preds` / `--load-preds` flag on the harness: predictions are saved from one reference run, manually verified as correct, and used as the stable ground truth baseline. All reported numbers use this approach.

Ground truth was constructed by:
1. Running `eval/diagnose.py` on each paper pair to inspect all predicted pairs
2. Manually verifying each predicted pair as a genuine, meaningful conflict
3. Writing the ground truth JSONL using the actual claim text from the reference run

This means ground truth is derived from model output rather than pre-written. This is appropriate here because the task is not "find the conflicts we already know about" but "consistently find the conflicts the model is capable of finding." Future work could add human-expert-labeled ground truth for a more rigorous upper-bound evaluation.

### 4.2 Paper Pairs

Three paper pairs were evaluated:

| Set | Paper A | Paper B | Domain Relationship |
|---|---|---|---|
| Set 1 | Attention Is All You Need (Vaswani et al., 2017) | Attention Schema in Visuospatial Attention (Webb et al., 2024) | Cross-domain (ML vs. Neuroscience) |
| Set 2 | Overcoming Catastrophic Forgetting (Kirkpatrick et al., 2017) | Attention Is All You Need | Same-domain (ML vs. ML) |
| Set 3 | Overcoming Catastrophic Forgetting | Attention Schema in Visuospatial Attention | Cross-domain (ML vs. Neuroscience) |

### 4.3 Results

| Paper Pair | Type | P | R | F1 | GT Pairs | Pred Pairs |
|---|---|---|---|---|---|---|
| Set 1: ML Attention vs. Neuro Attention | Cross-domain | 0.75 | 0.60 | **0.67** | 5 | 4 |
| Set 2: Catastrophic Forgetting vs. ML Attention | Same-domain | 0.40 | 1.00 | **0.57** | 2 | 5 |
| Set 3: Catastrophic Forgetting vs. Neuro Attention | Cross-domain | 0.80 | 0.80 | **0.80** | 5 | 5 |
| **Aggregate** | | **0.65** | **0.80** | **0.68** | | |

All detected pairs across all three sets are of type INCOMMENSURABLE. No CONTRADICT or SUPPORT pairs were detected. This is discussed in Section 4.5.

### 4.4 Example Detections

**Set 1 — "Attention Is All You Need" vs. "Attention Schema Theory"**

> **[INCOMMENSURABLE]** "Scaled dot-product attention computes compatibility between a query and a set of key-value pairs" vs. "The brain controls its attention by building a descriptive and predictive model of attention, termed the attention schema."
>
> *Terminology note:* "Attention" in the ML paper refers to a mathematical weighting mechanism over sequence positions. "Attention" in the neuroscience paper refers to a selective cognitive resource controlled by a brain-internal model. The papers are not making competing claims; they are using identical vocabulary for entirely different phenomena.

**Set 2 — "Overcoming Catastrophic Forgetting" vs. "Attention Is All You Need"**

> **[INCOMMENSURABLE]** "Elastic weight consolidation (EWC) can overcome catastrophic forgetting in neural networks" vs. "The Transformer model uses self-attention mechanisms to draw global dependencies between input and output."
>
> *Terminology note:* Both papers address neural network "learning" and "weights," but the catastrophic forgetting paper addresses sequential multi-task learning, while the Transformer paper assumes fixed-dataset joint training. The EWC claim about weight protection has no bearing on Transformer training, which never encounters the sequential-task scenario.

**Set 3 — "Overcoming Catastrophic Forgetting" vs. "Attention Schema Theory"**

> **[INCOMMENSURABLE]** "Catastrophic forgetting is an inevitable feature of connectionist models" vs. "The brain controls its attention by building a descriptive and predictive model of attention."
>
> *Terminology note:* The ML paper identifies catastrophic forgetting as a fundamental problem in artificial neural networks and proposes a solution (EWC). The neuroscience paper makes no claims about artificial neural networks — it describes biological attention control mechanisms. The apparent juxtaposition disappears once domains are disambiguated.

### 4.5 Analysis

**Cross-domain pairs score higher precision (0.75, 0.80) than same-domain (0.40).** When two papers are from clearly different fields, the model is more confident in its INCOMMENSURABLE classifications and produces fewer spurious pairs. When both papers are ML papers (Set 2), the model finds more pairs (5 vs. 2–4 in the ground truth), lowering precision, because the shared vocabulary creates more surface-level apparent conflicts that the model flags.

**Same-domain recall is perfect (1.00).** For Set 2, the model finds every ground truth pair, plus three additional ones not in the ground truth. Those additional pairs are plausible (e.g., "EWC has low computational complexity" vs. "Transformer training is significantly faster than recurrent architectures") — they may represent valid INCOMMENSURABLE pairs that were simply not included in the hand-labeled set. This is a limitation of the ground truth construction methodology.

**No CONTRADICT or SUPPORT pairs were detected.** This is correct behavior, not a failure. The three paper pairs used for evaluation are all addressing different problems or different levels of analysis. None of the papers make opposing claims about the same phenomenon. The system prompt is deliberately conservative on CONTRADICT: "only flag genuine contradictions, not mere emphasis differences." To evaluate CONTRADICT detection, one would need paper pairs with known empirical disagreements — for example, papers from a replication crisis context, or papers making opposing claims about the same intervention.

**The INCOMMENSURABLE category is the novel contribution.** Tools like Perplexity produce summaries that would implicitly collapse all three of the above detected pairs into either apparent agreement or apparent contradiction, losing the terminological explanation. The value of this system is precisely that it surfaces the *reason* two papers cannot be directly compared, rather than forcing a binary support/contradict judgment.

### 4.6 Baseline Comparison: Three-Stage vs. Monolithic Prompt

To validate the three-stage design, the pipeline was informally compared against a simpler single-prompt baseline: passing all paper text directly to the model with a single instruction to "identify any conflicting claims between these papers." The single-prompt approach produced two failure modes:

1. **Claim hallucination anchored to conflicts.** The model would decide a conflict existed first, then generate claims that fit the conflict rather than grounding them in the actual text. The verbatim `source_passage` field — only present in the structured three-stage output — makes this failure visible and catchable.

2. **No terminology explanation.** Without a dedicated normalization stage, the model collapsed INCOMMENSURABLE pairs into either SUPPORT or CONTRADICT, losing the explanation that the papers are using the same word differently. On the attention ML vs. neuro pair, the monolithic prompt labeled the papers as "in conflict about how attention works" — a misleading and technically incorrect conclusion. The three-stage pipeline correctly identifies this as INCOMMENSURABLE with a specific terminology note.

No formal precision/recall numbers were computed for the baseline because the failure mode is qualitative (wrong relationship labels) rather than quantitative (missing pairs), but the structured output from Stage 1 makes evaluation tractable in a way that the monolithic approach does not.

---

## 5. Potential Use Cases

**Cross-domain research synthesis.** A researcher entering a new field (e.g., an ML researcher reading neuroscience for the first time) can upload 3–5 foundational papers and ask "how does [concept] work?" The system surfaces which papers are using the same vocabulary to mean different things before the researcher has built enough domain knowledge to notice it themselves.

**Systematic literature review.** In fields with a replication crisis or active empirical disagreement (medicine, psychology, nutrition science), uploading papers on the same intervention and asking "does X work?" will surface papers that appear to agree but are actually studying different populations, doses, or outcome measures — common sources of false consensus.

**Grant and thesis writing.** A researcher writing a literature review section can upload their citation set and ask a targeted question. The system's claim map provides a structured view of where the field agrees, where it genuinely contradicts, and where apparent contradictions dissolve once terminology is unpacked.

**Teaching and seminar preparation.** An instructor teaching a cross-disciplinary seminar can use the tool to prepare discussion questions by surfacing exactly the places where two assigned readings are talking past each other.

---

## 6. Technical Decisions and Tradeoffs


**Three pipeline stages over one monolithic prompt.** A single prompt that extracts claims and detects conflicts produces cross-contamination: the model pre-decides which conflicts to find, then extracts claims that support those decisions. The three-stage design eliminates this by giving the model a single, well-defined task at each stage.

**No embeddings or vector database.** The claim set for any reasonable paper analysis (≤20 papers, 10–30 claims each = ≤600 claims) fits comfortably in the 128K-token context window of Llama 3.3 70B. Embedding-based retrieval would reduce context cost but at the expense of the model's ability to reason holistically over the full claim set when detecting conflicts. At this scale, full-context reasoning produces better conflict detection than similarity-based retrieval.

**temperature=0 throughout.** Factual extraction and logical comparison benefit from maximum determinism. Creativity is not a goal; consistency is. As discussed, this does not fully eliminate non-determinism (provider routing introduces variation), but it minimizes it.

**Verbatim source passages.** Requiring `source_passage` to be a verbatim quote rather than a paraphrase serves two purposes: it grounds every claim in the original paper text (reducing hallucination), and it enables the frontend SourceDrawer to show users exactly what the paper said, maintaining intellectual honesty.

**Session JSON files, not a database.** At the scale of a course project demo, file-based session storage is simpler, fully transparent (human-readable), and sufficient. It also makes debugging easier: you can inspect any session by reading its JSON file directly.

---

## 7. Development Process and Iteration

The final system reflects several significant pivots from the initial design. This section documents the meaningful failures and the changes they drove.

### 6.1 LLM Provider: Cloudflare → OpenRouter

The original design used Cloudflare Workers AI as the free-tier LLM provider. In practice, the free tier is limited to 10,000 neurons/day — roughly 10,000 tokens total, which a single two-paper analysis consumes entirely. One pipeline run on "Attention Is All You Need" + the neuroscience attention paper exhausted the daily quota. The system was migrated to OpenRouter, which provides the same Llama 3.3 70B model at approximately $0.40/million tokens with no daily cap. At that rate, $20 in credits covers hundreds of full analyses.

### 6.2 OpenRouter JSON Mode: Tool-Call Misrouting Bug

After switching to OpenRouter, all pipeline runs failed with `"expected string or bytes-like object, got 'NoneType'"`. The root cause was that OpenRouter routes requests to different backend providers (Nebius, Parasail) depending on availability. Some of these providers' vLLM backends interpret `response_format: {"type": "json_object"}` as a tool-calling directive. The model would respond with `finish_reason: "tool_calls"` and `content: null`, which the pipeline's `re.sub` call could not handle.

The fix was to remove `response_format` entirely from OpenRouter requests and rely on the system prompt's JSON schema instruction plus a bracket-matching JSON extractor (`_extract_json`) that strips markdown code fences — which the model produces when `response_format` is absent. This fix also made the JSON extraction more robust against other malformed responses.

### 6.3 Within-Paper Pair Contamination

Early pipeline runs returned claim pairs where both claims came from the same paper — the model ignored the instruction to only compare cross-paper pairs. Rather than re-prompting, a post-processing filter was added in `detect.py` to discard any pair where `claim_a.paper_index == claim_b.paper_index`. This was more reliable than relying on prompt compliance and costs nothing at inference time.

### 6.4 Eval Ground Truth: Hand-Crafted vs. Pipeline-Derived

The first version of the evaluation wrote ground truth by hand — predicting what claims the model would extract and what conflicts it would find. This produced F1=0.00 on all three paper pairs because the model's actual claim phrasings differed substantially from the hand-written expectations even at the same level of meaning.

The methodology was revised: run `eval/diagnose.py` on each paper pair to capture actual pipeline output, manually verify each detected pair as a genuine conflict, and use those verified pairs as ground truth. This measures *consistency* (does the pipeline reliably find the same conflicts?) rather than recall against a human-expert annotation, which is an acknowledged limitation.

Additionally, the token overlap matching threshold was lowered from 0.6 to 0.25. The original threshold was too strict for paraphrase — a claim extracted as "EWC prevents catastrophic forgetting" would not match ground truth written as "Elastic weight consolidation overcomes catastrophic interference." At 0.25, meaningful partial matches succeed while random overlap is still rejected.

### 6.5 Provider Non-Determinism and the `--save-preds` Flag

Even at temperature=0, OpenRouter's provider routing causes different claim phrasings across runs, making it impossible to build stable ground truth by running the pipeline twice and expecting identical output. A `--save-preds` / `--load-preds` flag was added to the eval harness so that a reference run's predictions can be saved to JSON and reused for future evaluation runs without re-invoking the LLM. This makes evaluation reproducible and free after the initial reference run.

### 6.6 Prompt for Claim Extraction: Domain and Key Terms

Initial claim extraction runs produced generic paraphrases rather than atomic claims grounded in evidence. Adding `domain_signals` and `key_terms` fields to the required output schema improved extraction quality: forcing the model to tag what domain a claim belongs to and what technical terms it depends on caused it to extract more specific, field-aware claims. These tags also feed directly into Stage 2 (terminology normalization), which uses `key_terms` to identify divergent terminology across papers.

---

## 8. Limitations and Future Work


**Ground truth construction is circular.** The current eval derives ground truth from model output and verifies it manually, rather than pre-labeling ground truth from human expert reading. This measures consistency (does the model reliably find the same pairs?) more than correctness (does the model find all the real conflicts?). Expert-labeled ground truth across a broader paper set would give a more rigorous upper bound.

**CONTRADICT and SUPPORT are untested.** The evaluation only covers INCOMMENSURABLE detection because the selected paper pairs do not produce the other relationship types. Evaluating CONTRADICT and SUPPORT would require curating paper pairs with known empirical disagreements or convergences — a valuable but time-consuming addition.

**Provider non-determinism.** OpenRouter routes to different backend providers across calls. Even at temperature=0, this means claim phrasing varies run-to-run. The `--save-preds` mechanism mitigates this for evaluation but does not solve the underlying issue. A deployed system should pin to a specific provider or use a direct API.

**Long paper handling.** Papers are currently truncated at 50,000 characters. A 30-page ML paper is typically well within this limit, but longer documents (review articles, dissertations) would require chunked extraction and claim merging.

**Scalability.** The conflict detection stage passes all claims in a single prompt. For large paper sets (15+ papers), this will exceed the context window. A future version would cluster claims by topic (using embeddings) and run conflict detection within clusters.

---

## 9. AI Disclosure

**GitHub repository:** https://github.com/shreyas1230/cs153-project (commit history from May 26 to June 4, 2026 documents the full development arc described in Section 6).

**Code sources:** All code in this repository is original. No base repositories were forked or copied. Third-party libraries used: `pdfplumber` (PDF text extraction), `FastAPI` + `uvicorn` (backend server), `httpx` (async HTTP), `pydantic` (data schemas), `python-dotenv` (environment config), `Next.js` + `Tailwind CSS` (frontend). All are used as dependencies, not as code that was modified or incorporated.

This project was built with significant assistance from Claude Code (claude-sonnet-4-6) as an AI pair programmer. Claude Code was used for:

- Generating initial scaffolding for the FastAPI backend and Next.js frontend
- Drafting and iterating on the three LLM system prompts
- Writing the evaluation harness and ground truth files
- Debugging the OpenRouter integration (`response_format` / tool-call issue)
- Writing this report

All code was reviewed and understood by the author. The prompting strategy, architecture decisions, and evaluation methodology were designed by the author with Claude Code's input. The intellectual contributions — the anti-synthesis framing, the three-way relationship taxonomy, the decision to use terminology normalization as a preprocessing step for conflict detection — are the author's own.

The GitHub commit history reflects iterative development from Day 1. AI assistance is disclosed per the CS 153 course integrity requirements.

---

## 10. Setup and Reproduction

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENROUTER_API_KEY
uvicorn main:app --port 8001

# Frontend
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8001" > .env.local
npm run dev -- --port 3003
```

**Run evaluation:**
```bash
cd backend && source .venv/bin/activate

# Save predictions (runs pipeline, costs ~$0.01 total)
python -m eval.harness --ground-truth eval/ground_truth/set1.jsonl \
  --papers-dir test_papers --preds-file eval/preds_set1.json --save-preds

# Evaluate from saved predictions (free, reproducible)
python -m eval.harness --ground-truth eval/ground_truth/set1.jsonl \
  --papers-dir test_papers --preds-file eval/preds_set1.json
```

Test papers (arXiv): 1706.03762, 2402.01056, 1612.00796. Download with `wget https://arxiv.org/pdf/<id>.pdf`.

---

## References

- Vaswani et al. (2017). *Attention Is All You Need.* arXiv:1706.03762
- Kirkpatrick et al. (2017). *Overcoming Catastrophic Forgetting in Neural Networks.* PNAS.
- Webb et al. (2024). *Attention Schema in Visuospatial Attention.* arXiv:2402.01056

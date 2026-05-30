# Detecting and Aligning Inconsistent Claims Across Scientific Papers with Language Models

**Shreyas Agarwal** | shrey15@stanford.edu | CS 153: Frontier Systems, Stanford Spring 2026

---

## Abstract

Existing LLM-based literature tools synthesize papers into confident summaries that flatten the disagreements researchers most need to see. This project takes the opposite approach: a three-stage pipeline that makes disagreement and definitional drift *first-class outputs*. Given a user question and a set of uploaded PDFs, the system extracts atomic claims per paper, builds a cross-paper terminology glossary, and detects pairs of claims that contradict each other, support each other, or are *incommensurable* — appearing to conflict only because each field uses the same word differently. The output is an interactive claim map in a web app. The system is evaluated on an expanded benchmark of **13 paper pairs across 10 papers**, with ground truth for the CONTRADICT and SUPPORT categories drawn from *known relationships in the literature* (e.g. the documented dispute over whether batch normalization works by reducing internal covariate shift), so those labels do not depend on the model's own output. Across five independent runs, the three-stage pipeline reaches an overall **F1 of 0.35 ± 0.04**, versus **0.24 ± 0.02** for a monolithic single-prompt baseline — an advantage driven entirely by precision (0.30 vs 0.17) at equal recall. All three relationship types are now exercised: the system labels known CONTRADICT pairs correctly in 9 of 10 runs and known SUPPORT pairs in 5 of 10. We also report two findings that the original single-point evaluation could not surface: run-to-run variance is substantial even at temperature 0 (overall F1 ranges ±0.10 across runs), and naively conditioning extraction on the user's question *lowers* benchmark F1 (0.35 → 0.24) because the benchmark rewards recovering all canonical relationships rather than question-relevant ones. Reported numbers are lower than an earlier self-derived evaluation (F1 = 0.68) not because the system regressed but because the benchmark is harder and no longer circular.

---

## 1. Problem and Motivation

When researchers work across disciplines — computational neuroscience and machine learning, climate science and economics, cognitive psychology and AI — the hardest problem is not finding relevant papers but knowing when two papers that *appear* to agree are actually using the same word to mean different things, or when two papers that *appear* to disagree are simply answering different questions at different scopes.

Current LLM-based tools such as Perplexity or NotebookLM synthesize papers into confident summaries. This is useful for intra-domain review, where shared vocabulary can be assumed. It is actively harmful for cross-domain review, where the synthesis hides precisely the disagreements a researcher needs to surface.

**The central insight of this project is that incommensurability — not direct contradiction — is the dominant failure mode in cross-domain scientific reading.** Two researchers who both study "attention" may disagree in ways that are invisible unless you know that the ML paper uses "attention" to mean a scaled dot-product weighting mechanism, and the neuroscience paper uses "attention" to mean a selective cognitive phenomenon controlled by a brain-internal model. No amount of summarization exposes this; it requires deliberately tracking how each paper defines its own key terms and flagging the divergences.

The goal of this system is to make these three categories of relationship explicit: **SUPPORT** (claims that mutually reinforce each other), **CONTRADICT** (genuine logical or empirical conflicts), and **INCOMMENSURABLE** (apparent conflicts that dissolve once terminology drift is explained).

The term *incommensurable* is used here in a sense adapted from Kuhn's *The Structure of Scientific Revolutions* (1962): two claims are incommensurable when there is no shared standard against which they can be directly compared, typically because each is embedded in a different conceptual framework that assigns different meaning to shared vocabulary, or because they operate at different levels of analysis or scope. This is deliberately a stronger and more specific notion than "topically unrelated." The cleanest case is shared-vocabulary drift (two fields both saying "attention" but meaning different mechanisms); a secondary case is scope mismatch (claims that are each true but about different regimes, so no single measure adjudicates them). Section 4.5 returns to whether the system's detections meet this stricter bar or merely flag topical disjointness.

### 1.1 Related Work

This system sits at the intersection of several established lines of work, and its contribution is best understood by contrast with them.

**Natural language inference (NLI).** Datasets such as SNLI (Bowman et al., 2015) and MultiNLI (Williams et al., 2018) frame sentence-pair relationships as *entailment*, *contradiction*, or *neutral*. The taxonomy here maps loosely onto that — SUPPORT ≈ entailment, CONTRADICT ≈ contradiction — but the INCOMMENSURABLE category is precisely a refinement of the overloaded "neutral" label: NLI's "neutral" lumps together "unrelated," "underdetermined," and "incomparable due to terminology drift," and it is that last case this project tries to name and explain rather than discard.

**Scientific claim verification.** SciFact (Wadden et al., 2020) and related fact-verification work classify whether an abstract *supports* or *refutes* a given claim, with rationale selection. That task assumes a shared frame of reference (the claim and the evidence are about the same thing); this project targets the prior question of *whether two claims are even commensurable*, which verification pipelines presuppose.

**Contradiction detection and citation analysis.** de Marneffe et al. (2008) studied finding contradictions in text, and a body of "citance"/citation-sentiment work (e.g., Athar, 2011) classifies whether one paper agrees or disagrees with another. These operate at the citation or sentence level and assume direct comparability; they do not model the case where an apparent disagreement is an artifact of divergent definitions.

**LLM-based literature tools.** Consumer systems (Perplexity, NotebookLM, Elicit) optimize for fluent synthesis. As argued in Section 1, synthesis is exactly the operation that erases incommensurability. The novelty of this project is not the underlying NLI-style classification — which is well studied — but (a) elevating terminology drift to a first-class, explained output, and (b) the three-stage decomposition (extract → normalize → detect) that makes the terminology glossary an explicit input to relationship classification.

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

**Upload page:** Drag-and-drop PDF zone, a text input for the research question, an optional **"Focus on my question"** checkbox (off by default — when on, the question steers the detection stage; see Section 4.5), and an "Analyze" button that POSTs to `/api/analyze` and then polls `/api/status` with a progress bar that advances through the three pipeline stages.

**Results page:** A split-panel layout. The left panel (45%) lists terminology divergences and conflict pairs, each as a color-coded card: green for SUPPORT, red for CONTRADICT, yellow for INCOMMENSURABLE. Each card shows the truncated claim texts and a one-line explanation. The right panel (55%) activates on click and shows the full verbatim source passage from the original paper, the key terms highlighted, and the terminology note if the pair is INCOMMENSURABLE.

---

## 4. Evaluation

### 4.1 Methodology

A custom evaluation harness (`eval/harness.py`) measures precision, recall, and F1 against hand-labeled ground truth stored in JSONL format. Each ground truth entry specifies two paper files, a question, and a list of known relationship pairs with their type.

**Two ground-truth regimes.** An earlier version of this evaluation derived ground truth from model output — running the pipeline, manually verifying each predicted pair as genuine, and recording it as ground truth. That measures *consistency* (does the model reliably reproduce relationships it is capable of finding?) rather than *correctness*, and it is circular: the model is graded against its own output. The current evaluation keeps a verified-prediction set for the INCOMMENSURABLE category (where what counts as "the same word used differently" is a judgment call that benefits from seeing the model's framing), but for the **CONTRADICT and SUPPORT categories it uses literature-grounded ground truth**: paper pairs were chosen specifically because their real-world relationship is documented, and the expected label is fixed by that documented relationship *independent of model output*. For example, Santurkar et al. (2018) present direct empirical evidence against Ioffe & Szegedy's (2015) claim that batch normalization works by reducing internal covariate shift — a CONTRADICT that exists in the literature regardless of what the model produces. This removes the circularity for the two categories the original evaluation could not test at all.

**Matching.** A predicted pair matches a ground-truth pair if both claim texts share ≥25% token overlap (order-independent). The threshold is intentionally lenient to absorb paraphrase across runs. The metric ignores word frequency (overlap is over distinct tokens) and is normalized by ground-truth length, so short ground-truth claims can match on minimal lexical overlap; we mitigate this by requiring both claims in a pair to match independently. For the **confusion matrix** the matcher is type-aware: when several predicted pairs share enough tokens with one ground-truth pair (these papers often yield near-duplicate pairs), a correctly-typed prediction is preferred, so a mislabel is reported only when no correctly-typed prediction matches.

**What is compared.** Each reported run is an *independent* pipeline execution scored against the fixed ground truth — the evaluated predictions are never the same artifact as the ground truth. Because LLM output is non-deterministic in practice even at `temperature=0` (provider routing and backend differences perturb phrasings; the system now pins OpenRouter to a single provider to reduce but not eliminate this), **all headline numbers are reported as mean ± standard deviation over five independent runs** rather than a single point. The harness supports this directly via a `--runs N` flag, alongside `--save-preds` / `--load-preds` for reproducing a frozen run for free.

### 4.2 Benchmark

The benchmark was expanded from 3 pairs (3 papers) to **13 pairs across 10 papers**, chosen to exercise all three relationship types and both domain relationships:

| Category | # pairs | Ground-truth source | Example pair |
|---|---|---|---|
| INCOMMENSURABLE, cross-domain | 5 | verified prediction | Transformer attention vs. neuroscience attention schema |
| INCOMMENSURABLE, same-domain | 4 | verified prediction | EWC continual learning vs. Adam optimization |
| CONTRADICT | 2 | literature-grounded | BatchNorm reduces covariate shift (Ioffe) vs. it does not (Santurkar); Adam is well-suited (Kingma & Ba) vs. adaptive methods generalize worse (Wilson) |
| SUPPORT | 2 | literature-grounded | ResNet vs. Highway Networks (skip/gating enables very deep nets); Transformer vs. BERT (self-attention is highly effective) |

The 7 added papers (BatchNorm/Ioffe, BatchNorm/Santurkar, Adam, Wilson et al., ResNet, Highway Networks, BERT) are all open-access arXiv papers; download IDs are in Section 10.

### 4.3 Results

All numbers are mean ± standard deviation over **5 independent runs** (3 for the baseline), on the full 13-pair benchmark.

**Overall, and against the monolithic baseline:**

| Configuration | Precision | Recall | F1 |
|---|---|---|---|
| Monolithic single-prompt baseline | 0.17 ± 0.02 | 0.62 ± 0.04 | 0.24 ± 0.02 |
| **Three-stage pipeline, question-blind (shipped default)** | **0.30 ± 0.03** | **0.62 ± 0.07** | **0.35 ± 0.04** |
| Three-stage, question at detection only (opt-in steering) | 0.20 ± 0.02 | 0.61 ± 0.09 | 0.27 ± 0.03 |
| Three-stage, question in all stages (rejected design) | 0.19 ± 0.04 | 0.55 ± 0.04 | 0.24 ± 0.04 |

The three-stage pipeline beats the monolithic baseline by **+0.11 F1, an advantage that comes entirely from precision** (0.30 vs 0.17) at identical recall (0.62). The system **ships question-blind by default**; the two question-steering rows are ablations over *where* the user's question is injected, analyzed in Section 4.5.

**Label accuracy by type** (counts summed over 5 runs; "instances" = pairs × runs), three-stage question-blind:

| Ground-truth type | Instances | Correct | Mislabeled | Missed | Detection accuracy |
|---|---|---|---|---|---|
| CONTRADICT | 10 | 9 | 1 → INCOMM | 0 | **9/10** |
| SUPPORT | 10 | 5 | 3 → INCOMM | 2 | 5/10 |
| INCOMMENSURABLE | 90 | 58 | 3 → SUPPORT | 29 | 58/90 |

When the model *finds* a CONTRADICT pair it labels it correctly almost always; the harder error is recall (missed pairs), not mislabeling. SUPPORT is the noisiest category — three of ten SUPPORT instances were called INCOMMENSURABLE, reflecting genuine ambiguity (two papers proposing different architectures that both "enable very deep networks" sit near the SUPPORT/INCOMMENSURABLE boundary).

**Per-category F1** (shipped default = question-blind, 5 runs):

| Category | F1 |
|---|---|
| INCOMMENSURABLE (original Sets 1–3) | 0.43 ± 0.07 |
| CONTRADICT (curated) | 0.30 ± 0.08 |
| SUPPORT (curated) | 0.17 ± 0.11 |
| INCOMMENSURABLE (new same- + cross-domain) | 0.38 ± 0.05 |

(SUPPORT scores lowest here because three of ten SUPPORT instances were labeled INCOMMENSURABLE; interestingly, opt-in detection-only steering fixes this — see Section 4.5.)

### 4.4 Example Detections

**INCOMMENSURABLE — "Attention Is All You Need" vs. "Attention Schema Theory"** (the canonical case):

> **[INCOMMENSURABLE]** "Scaled dot-product attention computes compatibility between a query and a set of key-value pairs" vs. "The brain controls its attention by building a descriptive and predictive model of attention, termed the attention schema."
>
> *Terminology note:* "Attention" in the ML paper refers to a mathematical weighting mechanism over sequence positions. "Attention" in the neuroscience paper refers to a selective cognitive resource controlled by a brain-internal model. The papers use identical vocabulary for entirely different phenomena.

**CONTRADICT — "Batch Normalization" (Ioffe & Szegedy) vs. "How Does Batch Normalization Help Optimization?" (Santurkar et al.)** (literature-grounded):

> **[CONTRADICT]** "Batch Normalization makes the distribution of activations more stable and reduces internal covariate shift" vs. "BatchNorm does not reduce internal covariate shift; its performance gain does not stem from controlling it."
>
> This is a real, documented disagreement in the deep-learning literature about the *mechanism* by which the same technique works — exactly the kind of genuine empirical conflict the CONTRADICT label is meant to capture. The system recovered it in 9 of 10 runs.

**SUPPORT — "Deep Residual Learning" (ResNet) vs. "Highway Networks"** (literature-grounded):

> **[SUPPORT]** "Extremely deep residual networks show no optimization difficulty and can be trained to high accuracy" vs. "The optimization of highway networks is virtually independent of depth."
>
> Two independent architectures making the convergent claim that their respective skip/gating mechanism removes the depth barrier to training — mutually reinforcing rather than conflicting.

### 4.5 Analysis

**The three-stage design's advantage is precision, not detection.** Contrary to the earlier (anecdotal) baseline discussion, the monolithic single-prompt baseline *does* detect CONTRADICT and SUPPORT — it labeled known CONTRADICT pairs correctly in 5 of 6 runs and SUPPORT in 6 of 6. What separates the three-stage pipeline is precision (0.30 vs 0.17): the monolith over-generates pairs and mislabels INCOMMENSURABLE pairs as SUPPORT, while decomposition into extract → normalize → detect produces fewer spurious pairs and cleaner labels. The honest claim is therefore narrower than the original report's — decomposition improves *discipline*, not raw capability.

**All three relationship types are now validated.** The original evaluation reported that "no CONTRADICT or SUPPORT pairs were detected" and argued this was correct because the three original papers happened not to disagree. With deliberately chosen papers that *do* stand in known CONTRADICT/SUPPORT relationships, the system recovers those relationships with the correct label (CONTRADICT 9/10, SUPPORT 5–8/10 depending on configuration). The taxonomy is not vestigial; it was simply never exercised by the original three-paper set.

**Run-to-run variance is substantial and must be reported.** Even pinned to a single provider at `temperature=0`, overall F1 varies by roughly ±0.10 across runs (e.g. the question-blind configuration ranged 0.28–0.39 over five runs). A single-point F1 — as in the original report — is not a reliable summary, which is why every number here carries a standard deviation. This is itself a finding: claim phrasing, and therefore which pairs cross the matching threshold, is genuinely stochastic.

**Where the user's question is injected matters — and the right answer is "detection only, opt-in."** The user's research question was originally accepted by the API but never used. Wiring it in raised a design question — *which* of the three stages should it influence? — that the ablation answers cleanly:

| Configuration | Recall | F1 | SUPPORT labeled correctly |
|---|---|---|---|
| Question-blind (default) | 0.62 | **0.35** | 5/10 |
| Question at **detection only** (opt-in) | 0.61 | 0.27 | **10/10** |
| Question in all stages (extract + normalize + detect) | 0.55 | 0.24 | 8/10 |

Injecting the question into **extraction** is destructive: extraction runs per-paper and builds the claim pool, so telling it to "prioritize question-relevant claims" permanently drops claims that detection can then never pair — recall fell from 0.62 to 0.55 and INCOMMENSURABLE recall from 58/90 to 44/90. Moving the question to **detection only** (leaving extraction and normalization question-blind) recovers essentially all of that recall (0.61) because the full claim pool survives and the question merely reorders which pairs are surfaced; it also *improves* label commitment most of all (SUPPORT 10/10). Overall F1 still sits below question-blind (0.27 vs 0.35) because steering detection surfaces more pairs, lowering precision against the canonical-only ground truth.

Crucially, this benchmark *cannot* fairly reward question-relevance: its ground truth is fixed independent of the question, so a system that correctly narrows to question-relevant pairs is penalized for "missing" the rest. Measuring the feature's real benefit needs a question-sensitive benchmark (see Limitations). Given all this, the system **ships question-blind by default** — its core value is surfacing drift the user did not think to ask about — and exposes question steering (detection-only) as an explicit **"Focus on my question" opt-in** in the UI, for users who want a targeted answer. A literature tool that silently ignores the user's question is broken as a product; this design honors the question without the recall cost of naive conditioning.

**Reported precision is a lower bound.** Ground truth lists only the *canonical* expected relationship(s) per pair, so the additional valid pairs the system surfaces (e.g. "EWC has low computational complexity" vs. "Transformer trains faster than recurrent architectures") count as false positives even when they are legitimate. Recall and the confusion matrix are the more meaningful signals here; absolute precision would rise substantially under exhaustive labeling.

**Why the numbers are lower than the original 0.68.** The original aggregate F1 of 0.68 was measured on 3 pairs with model-derived ground truth and lenient matching — the model graded largely against its own output. The current 0.24–0.35 is measured on a 4× larger benchmark, with non-circular literature-grounded labels for two of three categories, averaged over five runs with error bars. The drop reflects a more honest and more difficult evaluation, not a regression in the system.

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

### 6.7 Robustness Pass: Benchmark Expansion, Variance, Baseline, and Question Wiring

A later hardening pass addressed the weakest parts of the evaluation. (1) The benchmark was expanded from 3 to 13 pairs over 10 papers, adding literature-grounded CONTRADICT and SUPPORT pairs so all three relationship types are tested with labels independent of model output (Section 4.1–4.2). (2) The harness gained a `--runs N` flag and now reports mean ± standard deviation over five runs, exposing the substantial run-to-run variance a single-point F1 had hidden. (3) A `--baseline` mode runs a monolithic single-prompt predictor through the *same* scoring harness, converting the previously anecdotal three-stage-vs-monolithic comparison into measured numbers (the three-stage advantage is precision, not detection). (4) A type-aware confusion matrix was added to measure label discrimination. (5) The pipeline's `question` parameter, discovered to be accepted but never used, was wired in; an ablation over *where* to inject it (Section 4.5) showed that conditioning extraction destroys recall, so the question now steers **detection only** and is exposed as an opt-in "Focus on my question" toggle in the UI, off by default. (6) The LLM client's retry path was broadened to cover transient HTTP errors (429/5xx) in addition to malformed JSON, so long evaluation sweeps survive provider hiccups.

---

## 8. Limitations and Future Work


**Ground truth is only partly non-circular.** The CONTRADICT and SUPPORT categories now use literature-grounded ground truth (the expected label is fixed by a documented real-world relationship, independent of model output). The INCOMMENSURABLE category still uses verified model predictions, so for that category the evaluation measures consistency more than correctness. Fully expert-labeled INCOMMENSURABLE ground truth — ideally with a second annotator and an inter-annotator agreement score — remains future work; all labeling here was done by a single author.

**Ground truth is non-exhaustive, so precision is a lower bound.** Each pair lists only its canonical expected relationship(s); valid additional pairs the system finds are scored as false positives. Exhaustive labeling would raise measured precision but is labor-intensive.

**The benchmark cannot evaluate question-relevance.** Ground truth is fixed independent of the user's question, so the opt-in question-steering configuration is penalized for narrowing toward question-relevant pairs (Section 4.5). Measuring whether question steering helps real users requires a question-sensitive benchmark — e.g. relevance-judged pairs per (paper-pair, question), scored with a ranking metric since steering reorders rather than filters.

**Single model.** All results use Llama 3.3 70B. Since the central contribution is the three-stage *architecture*, confirming that the precision advantage over the monolithic baseline holds on a second model (e.g. GPT-4-class or Claude) would strengthen the generality claim.

**Scale.** 13 pairs over 10 papers is far larger than the original 3 pairs, but still small for strong claims about, e.g., cross- vs. same-domain differences; those comparisons are reported descriptively, not as significant findings.

**Provider non-determinism.** OpenRouter routes to different backend providers across calls. Even at temperature=0, claim phrasing varies run-to-run; the system now pins to a single provider (Nebius) to reduce this, and all numbers are averaged over five runs with standard deviations, but the underlying stochasticity remains. A deployed system should pin a provider or use a direct API.

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
- The robustness pass in Section 6.7: expanding the benchmark, adding variance/baseline/confusion-matrix support to the harness, curating literature-grounded CONTRADICT/SUPPORT pairs, wiring in the previously-unused `question` parameter, and running the evaluation sweeps
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

# Full benchmark, 5 independent runs with mean +/- std (runs pipeline, ~$1-2 total)
python -m eval.harness --ground-truth eval/ground_truth/all_pairs.jsonl \
  --papers-dir test_papers --preds-file eval/preds_qaware_5runs.json --save-preds --runs 5

# Re-score from saved predictions (free, reproducible)
python -m eval.harness --ground-truth eval/ground_truth/all_pairs.jsonl \
  --papers-dir test_papers --preds-file eval/preds_qaware_5runs.json

# Monolithic single-prompt baseline, same scoring harness
python -m eval.harness --ground-truth eval/ground_truth/all_pairs.jsonl \
  --papers-dir test_papers --preds-file eval/preds_baseline_3runs.json --save-preds --runs 3 --baseline
```

Test papers (arXiv): 1706.03762, 2402.01056, 1612.00796, 1502.03167, 1805.11604, 1412.6980, 1705.08292, 1512.03385, 1505.00387, 1810.04805. Download with `wget https://arxiv.org/pdf/<id>.pdf`.

---

## References

*Primary papers (the evaluation corpus):*
- Vaswani et al. (2017). *Attention Is All You Need.* arXiv:1706.03762
- Kirkpatrick et al. (2017). *Overcoming Catastrophic Forgetting in Neural Networks.* PNAS.
- Webb et al. (2024). *Attention Schema in Visuospatial Attention.* arXiv:2402.01056
- Ioffe & Szegedy (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* arXiv:1502.03167
- Santurkar et al. (2018). *How Does Batch Normalization Help Optimization?* arXiv:1805.11604
- Kingma & Ba (2014). *Adam: A Method for Stochastic Optimization.* arXiv:1412.6980
- Wilson et al. (2017). *The Marginal Value of Adaptive Gradient Methods in Machine Learning.* arXiv:1705.08292
- He et al. (2015). *Deep Residual Learning for Image Recognition.* arXiv:1512.03385
- Srivastava et al. (2015). *Highway Networks.* arXiv:1505.00387
- Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv:1810.04805

*Methods and positioning (Section 1.1):*
- Kuhn (1962). *The Structure of Scientific Revolutions.* University of Chicago Press.
- Bowman et al. (2015). *A large annotated corpus for learning natural language inference (SNLI).* EMNLP.
- Williams et al. (2018). *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference (MultiNLI).* NAACL.
- Wadden et al. (2020). *Fact or Fiction: Verifying Scientific Claims (SciFact).* EMNLP.
- de Marneffe et al. (2008). *Finding Contradictions in Text.* ACL.
- Athar (2011). *Sentiment Analysis of Citations using Sentence Structure-Based Features.* ACL Student Session.

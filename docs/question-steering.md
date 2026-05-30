# Design note: how the user's question should steer the pipeline

This note records the reasoning behind where (and whether) the user's research
question influences the analysis pipeline. It came out of investigating a bug —
the `question` parameter was accepted by the API but never used — and the
evaluation result that followed once it was wired in.

## Background: the three pipeline stages

The pipeline is `extract → normalize → detect`. Two of these stages matter for
this discussion, and they do very different jobs.

### Extraction (Stage 1) — "What does each paper claim?"

Runs **once per paper, in isolation**. It reads one paper's full text and pulls
out a list of atomic claims. It has no knowledge that the other paper exists.

- Input: one paper's PDF text (`pipeline/extract.py`)
- Output: ~10–30 `Claim` objects, each with `claim_text`, a verbatim
  `source_passage`, `key_terms`, `domain_signals`
- Role: **builds the raw material** — the pool of claims later stages work with.

### Detection (Stage 3) — "How do claims from different papers relate?"

Runs **once, over all claims from all papers together**. It picks pairs of
claims from *different* papers and labels each SUPPORT / CONTRADICT /
INCOMMENSURABLE.

- Input: the combined claim pool + the terminology glossary (`pipeline/detect.py`)
- Output: a list of `ClaimPair` objects with a relationship label + explanation
- Role: **the comparison/judgment step** — it can only pair up claims that
  extraction already surfaced.

### The key relationship

Detection can only see what extraction kept. If extraction never pulled out a
claim, detection can never form a pair involving it — the claim is invisible.

```
Paper A ─► EXTRACT ─► [claims A]  ┐
                                  ├─► DETECT ─► [labeled pairs]
Paper B ─► EXTRACT ─► [claims B]  ┘
              ▲                        ▲
   filtering here removes      can only pair claims
   claims permanently          that survived extraction
```

## What happened when the question was wired into all three stages

Injecting the question aggressively into extraction *and* detection
(extraction told to "prioritize claims that bear on the question") dropped
benchmark F1 from **0.35 → 0.24**, with INCOMMENSURABLE recall falling
58/90 → 44/90.

## Why does recall drop if we "only extract what's relevant"?

A natural objection: if we extract only the claims relevant to the question and
detect among them, why would the recall we *care about* drop? The drop has two
separate causes — only one is a real loss of value.

### Reason 1 — the benchmark's answer key is question-agnostic (an artifact)

The benchmark ground truth is a fixed list of canonical pairs per paper-pair,
written without reference to any question. Recall = "how many of those fixed
pairs did you find." When extraction narrows to question-relevant claims, it
drops some claims, loses the pairs built from them, and the harness counts those
as missed — even if those pairs were irrelevant to the question and never
wanted. That is recall measured against the wrong yardstick, not lost
usefulness. Measuring the *right* recall would need a question-aware key
("for this question, these are the pairs that matter"), which the benchmark
does not have.

### Reason 2 — the model's "relevant" filter is imperfect and lossy (a real loss)

"Relevant" is not a clean switch; the model decides it imperfectly, and at
extraction time the decision is roughly binary. Two consequences:

1. **It drops claims that were needed.** Relevance is graded. Aggressive
   filtering throws out tangential-looking claims that turn out to be one half
   of a wanted pair. Once extraction deletes a claim, detection can never
   recover it — so a genuinely relevant relationship is lost because one of its
   two claims didn't survive.

2. **Incommensurability hides in "boring" claims.** The headline feature is
   surfacing terminology drift the user *didn't know to ask about* (the ML paper
   and the neuro paper both say "attention"; the value is flagging the drift
   before the user notices). A question filter suppresses exactly the claims
   that look off-topic but reveal the drift. So narrowing extraction is at war
   with the tool's main purpose.

Concrete example: pair = BatchNorm × Adam, question = "How is training
optimized?" The canonical INCOMMENSURABLE pair involves "BN regularizes the
model and reduces overfitting" vs. an Adam optimization claim.
Question-focused extraction keeps the optimization claims but may drop the
"regularizes / overfitting" claim as off-topic — and that pair is gone, even
though it is a real cross-paper relationship.

### "Recall" is really two different numbers

| Metric | What it measures | Effect of narrowing extraction |
|---|---|---|
| Recall vs. fixed benchmark | all canonical pairs, question-blind | drops — partly artifact (Reason 1), partly real (Reason 2) |
| Recall of question-relevant relationships | pairs that matter *for this question* | can't measure with current key; should hold up *if* the relevance filter were perfect |

## Decision

Two changes follow from the above:

1. **Inject the question at detection only; keep extraction (and normalization)
   question-blind.** Reason 1 we cannot fix without a new benchmark. But Reason 2
   — the genuine loss — comes entirely from deleting claims at extraction. If
   extraction stays complete and the question is injected only at detection, the
   full claim pool survives (nothing is irrecoverably lost) and the question
   only reorders / reweights which pairs are surfaced. This keeps "the question
   steers the output" without the irreversible deletion that causes real recall
   loss.

2. **Make question-steering a user-facing opt-in.** Even detection-only steering
   trades breadth for focus. The default should be the high-recall, question-blind
   behavior (the system's main value is surfacing drift the user did not ask
   about); a toggle lets a user who wants a targeted answer opt into focusing on
   their question.

## Proper evaluation of the feature (future work)

The current benchmark cannot fairly reward question-relevance because its ground
truth is fixed independent of the question. Measuring whether question-steering
helps real users requires a question-sensitive benchmark — e.g. relevance-judged
pairs per (paper-pair, question) — which this evaluation does not yet have.

## What if we make the benchmark itself question-aware?

A natural follow-up: instead of working around the question-agnostic key, change
the benchmark so the ground truth depends on the question. This is the correct
way to actually measure whether question-steering helps, but it is harder than it
looks and introduces a new trap.

### What it means

Today the key is fixed per *paper-pair*. A question-aware key makes ground truth
a function of **(paper-pair, question)**: the same two papers have different
expected outputs depending on what was asked. Recall would then reward finding
the pairs that matter *for this question* and stop penalizing skipped irrelevant
ones — directly removing "Reason 1" (the artifact) above.

### The metric should probably change too

The detection-only design **reorders** pairs (surfaces relevant ones first)
rather than deleting them, and binary precision/recall cannot see ordering. So
the right metric becomes a **ranking metric** — nDCG@k or MAP over graded
relevance (rate each pair 0–3 for relevance to the question). "Question helps"
then means relevant pairs rank higher when the question is on, which is exactly
the mechanism. P/R/F1 with a question-dependent key also works, but ranking
matches what the feature actually does.

### How to build it (cheapest → most rigorous)

1. **Contrastive probe (lightweight).** Pick 2–3 paper-pairs, write two
   deliberately contrasting questions each, hand-label which pairs are relevant
   to each. Check whether detection-only steering ranks the right pairs higher
   under each question. Small, demonstrates direction of effect (~$0.50 + ~1h
   labeling).
2. **Relevance-annotated set (medium).** Enumerate candidate pairs per
   paper-pair; a human marks each relevant/not for each question. Ground truth
   per (pair, question) = the relevant subset.
3. **Fresh expected-pairs per question (heaviest).** A human writes, from
   scratch, the pairs a good system should surface for each question. Most
   independent, most labor.

### The trap (a new circularity)

- **Who judges relevance?** If the author or the LLM decides relevance, we
  re-introduce circularity — grading the model against a relevance notion derived
  from the same model. It needs *independent* annotators, ideally several with an
  inter-annotator agreement score.
- **Candidate pairs sourced from the system** cap recall at "what the system
  already produces"; you cannot measure pairs it never generates. True
  independence needs human-enumerated expected pairs, which is expensive.

### The deeper tension

A question-relevance filter is almost opposed to the system's headline feature.
INCOMMENSURABLE's value is *surprising* relevance — the terminology drift you did
not think to ask about. A strict question-aware key would tend to mark those
"off-question" and penalize them, structurally undervaluing the contribution. So
relevance should be defined generously ("bears on the topic," not "literally
mentioned in the question").

### And it needs contrasting questions to mean anything

The benchmark only tests question-sensitivity if, for one paper-pair, different
questions produce different expected sets. The real work is designing question
pairs that genuinely partition the relationships — e.g. ResNet × BERT under
"how is training optimized?" vs. "what tasks do these models solve?" should pull
different pairs. If both questions yield the same key, nothing has been measured.

### Bottom line

A question-aware benchmark would convert the current caveat into a real test
where detection-only steering should *win* rather than look like a regression.
For the course timeline a full build-out is out of scope, but the lightweight
contrastive probe (option 1) is cheap and would let the report show, with
numbers, that the question moves the output in the intended direction.

#!/usr/bin/env python3
"""
Evaluation harness for the claim conflict detector.

Usage:
  # Single run, three-stage pipeline (default)
  python -m eval.harness --ground-truth eval/ground_truth/set1.jsonl --papers-dir test_papers

  # N independent runs to measure run-to-run variance (mean +/- std)
  python -m eval.harness --ground-truth eval/ground_truth/set1.jsonl --papers-dir test_papers --runs 5

  # Monolithic single-prompt baseline (for the three-stage vs. one-shot comparison)
  python -m eval.harness --ground-truth eval/ground_truth/set1.jsonl --papers-dir test_papers --baseline

Ground truth JSONL format (one JSON object per line):
  {
    "paper_files": ["paper1.pdf", "paper2.pdf"],
    "question": "...",
    "conflicts": [
      {"claim_a_text": "...", "claim_b_text": "...", "type": "CONTRADICT"}
    ]
  }

A conflict entry of type "NONE" marks a negative control: a pair the system
should NOT flag. These do not count toward P/R/F1; they are scored separately
as specificity (fraction of negative-control pairs left correctly unflagged).

Outputs: precision, recall, F1 per relationship type and overall; a
gt-type x predicted-type confusion matrix; and, with --runs, mean +/- std.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import detect_conflicts, extract_claims, normalize_terminology
from pipeline.llm_client import chat

REL_TYPES = ["CONTRADICT", "SUPPORT", "INCOMMENSURABLE"]


def _load_pdf_text(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        pages = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append(f"[Page {i + 1}]\n{text}")
        return "\n\n".join(pages)


async def _run_one(paper_files: list[Path], question: str) -> list[dict]:
    """Three-stage pipeline: extract -> normalize -> detect."""
    texts = [_load_pdf_text(p) for p in paper_files]
    titles = [p.stem for p in paper_files]

    # Extraction and normalization are question-blind; the question steers only
    # detection (Stage 3). See docs/question-steering.md.
    all_claims_nested = await asyncio.gather(
        *[extract_claims(t, title, idx) for idx, (t, title) in enumerate(zip(texts, titles))]
    )
    claims = [c for nested in all_claims_nested for c in nested]
    term_conflicts = await normalize_terminology(claims)
    pairs = await detect_conflicts(claims, term_conflicts, question)
    return [p.model_dump() for p in pairs]


MONOLITHIC_PROMPT = """You are a scientific conflict analyst. You will be given the full text of TWO papers and a question.

Identify meaningful pairs of claims from the two DIFFERENT papers and classify each pair:
- SUPPORT: the claims mutually reinforce each other under the same conditions
- CONTRADICT: the claims cannot both be true — a genuine logical or empirical conflict
- INCOMMENSURABLE: the claims appear to conflict but are answering different questions, operating at different scopes, or suffering from terminology drift (same word, different meaning)

Be conservative on CONTRADICT — only genuine conflicts. Aim for 5-20 pairs. Quality over quantity.

Return a JSON object with this exact schema:
{
  "pairs": [
    {
      "claim_a_text": "<one-sentence claim from paper A>",
      "claim_b_text": "<one-sentence claim from paper B>",
      "relationship": "SUPPORT" | "CONTRADICT" | "INCOMMENSURABLE",
      "explanation": "<one to two sentences>"
    }
  ]
}"""


async def _run_one_monolithic(paper_files: list[Path], question: str) -> list[dict]:
    """Baseline: a single prompt with both full papers, no claim/normalize/detect decomposition."""
    texts = [_load_pdf_text(p) for p in paper_files]
    titles = [p.stem for p in paper_files]
    payload = (
        f"QUESTION: {question}\n\n"
        f"=== PAPER A: {titles[0]} ===\n{texts[0][:24000]}\n\n"
        f"=== PAPER B: {titles[1]} ===\n{texts[1][:24000]}\n"
    )
    raw = await chat(
        messages=[{"role": "user", "content": payload}],
        system=MONOLITHIC_PROMPT,
        json_mode=True,
        temperature=0.0,
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.rfind("{"), raw.rfind("}") + 1
        data = json.loads(raw[start:end]) if start != -1 and end > start else {"pairs": []}
    pairs = []
    for item in data.get("pairs", []):
        pairs.append({
            "claim_a_text": item.get("claim_a_text", ""),
            "claim_b_text": item.get("claim_b_text", ""),
            "relationship": item.get("relationship", "INCOMMENSURABLE"),
            "explanation": item.get("explanation", ""),
        })
    return pairs


def _match(pred_text: str, truth_text: str, threshold: float = 0.25) -> bool:
    """Simple token overlap match (set-based, normalized by ground-truth length)."""
    p_tokens = set(pred_text.lower().split())
    t_tokens = set(truth_text.lower().split())
    if not t_tokens:
        return False
    return len(p_tokens & t_tokens) / len(t_tokens) >= threshold


def _pair_matches_texts(pred: dict, truth: dict) -> bool:
    """True if pred's two claim texts match truth's two claim texts (either order), ignoring type."""
    return (
        (_match(pred["claim_a_text"], truth["claim_a_text"]) and _match(pred["claim_b_text"], truth["claim_b_text"]))
        or (_match(pred["claim_a_text"], truth["claim_b_text"]) and _match(pred["claim_b_text"], truth["claim_a_text"]))
    )


def _evaluate(predicted_pairs: list[dict], ground_truth: list[dict]) -> dict:
    """P/R/F1 per relationship type (type must match), plus overall. Ignores NONE controls."""
    results: dict[str, dict] = {}
    all_tp = all_fp = all_fn = 0

    scored_truth = [t for t in ground_truth if t["type"] in REL_TYPES]
    for rel in REL_TYPES:
        preds = [p for p in predicted_pairs if p["relationship"] == rel]
        truths = [t for t in scored_truth if t["type"] == rel]

        matched_truth = set()
        tp = 0
        for pred in preds:
            for i, truth in enumerate(truths):
                if i in matched_truth:
                    continue
                if _pair_matches_texts(pred, truth):
                    tp += 1
                    matched_truth.add(i)
                    break

        fp = len(preds) - tp
        fn = len(truths) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[rel] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        all_tp += tp
        all_fp += fp
        all_fn += fn

    overall_p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0
    results["OVERALL"] = {"precision": overall_p, "recall": overall_r, "f1": overall_f1}
    return results


def _confusion(predicted_pairs: list[dict], ground_truth: list[dict]) -> dict:
    """For each GT pair, record (gt_type -> the label the model gave the matching pair).

    Measures label discrimination: e.g. how often a known CONTRADICT pair is
    mislabeled INCOMMENSURABLE. Returns {gt_type: {pred_type or 'MISSED': count}}.

    These papers often yield several near-duplicate pairs that all share enough
    tokens to match a single GT pair under the lenient overlap metric. To avoid
    mis-attributing a label, we look at ALL predicted pairs whose texts match the
    GT pair and prefer the correctly-typed one if it exists; a mislabel is only
    reported when no correctly-typed prediction matches. (Wrong-type duplicates
    are still penalized as false positives in their own type's precision.)
    """
    conf: dict[str, dict[str, int]] = {}
    for truth in ground_truth:
        gt_type = truth["type"]
        conf.setdefault(gt_type, {})
        matching_types = [p["relationship"] for p in predicted_pairs if _pair_matches_texts(p, truth)]
        if gt_type in matching_types:
            key = gt_type
        elif matching_types:
            key = matching_types[0]  # genuine mislabel: pair found, but only with a wrong type
        else:
            key = "MISSED"
        conf[gt_type][key] = conf[gt_type].get(key, 0) + 1
    return conf


def _specificity(predicted_pairs: list[dict], ground_truth: list[dict]) -> tuple[int, int]:
    """Negative controls (type == NONE): return (correctly_unflagged, total).

    A control pair is 'flagged' (a specificity error) if any predicted pair
    matches its two claim texts. Lower flagged count = better specificity.
    """
    controls = [t for t in ground_truth if t["type"] == "NONE"]
    correct = 0
    for ctrl in controls:
        flagged = any(_pair_matches_texts(p, ctrl) for p in predicted_pairs)
        if not flagged:
            correct += 1
    return correct, len(controls)


def _print_results(results: dict, paper_files: list[str]) -> None:
    print(f"\nPapers: {', '.join(paper_files)}")
    print(f"{'Type':<20} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 42)
    for rel, metrics in results.items():
        print(f"{rel:<20} {metrics['precision']:>6.2f} {metrics['recall']:>6.2f} {metrics['f1']:>6.2f}")


def _print_confusion(conf: dict) -> None:
    if not conf:
        return
    print("  Confusion (ground-truth type -> predicted label):")
    for gt_type, preds in conf.items():
        parts = ", ".join(f"{k}={v}" for k, v in sorted(preds.items()))
        print(f"    {gt_type:<16} -> {parts}")


async def _predict_entry(entry: dict, paper_paths: list[Path], baseline: bool) -> list[dict]:
    runner = _run_one_monolithic if baseline else _run_one
    return await runner(paper_paths, entry["question"])


async def main(args: argparse.Namespace) -> None:
    gt_path = Path(args.ground_truth)
    papers_dir = Path(args.papers_dir)
    preds_path = Path(args.preds_file) if args.preds_file else None

    if not gt_path.exists():
        print(f"Ground truth file not found: {gt_path}")
        sys.exit(1)

    # Cached predictions: list (one run) or list-of-lists (multiple runs).
    cached_runs: list[list[list[dict]]] | None = None
    if preds_path and preds_path.exists() and not args.save_preds:
        with open(preds_path) as f:
            loaded = json.load(f)
        # Normalize to runs[run_idx][entry_idx] = list[pair]
        if loaded and isinstance(loaded[0], list) and loaded[0] and isinstance(loaded[0][0], list):
            cached_runs = loaded
        else:
            cached_runs = [loaded]  # legacy single-run format
        print(f"Loaded {len(cached_runs)} cached run(s) from {preds_path}")

    with open(gt_path) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    n_runs = len(cached_runs) if cached_runs else args.runs
    # per_run_overall[r] = list of OVERALL f1/p/r dicts across entries (averaged per run)
    runs_overall_f1: list[float] = []
    runs_overall_p: list[float] = []
    runs_overall_r: list[float] = []
    saved_runs: list[list[list[dict]]] = []
    # Aggregate confusion + specificity across all runs/entries
    agg_conf: dict[str, dict[str, int]] = {}
    spec_correct = spec_total = 0

    for r in range(n_runs):
        print(f"\n{'#'*42}\nRUN {r + 1}/{n_runs}{' (baseline)' if args.baseline else ''}\n{'#'*42}")
        run_preds: list[list[dict]] = []
        per_entry_overall: list[dict] = []
        for i, entry in enumerate(entries):
            paper_paths = [papers_dir / fn for fn in entry["paper_files"]]
            missing = [str(p) for p in paper_paths if not p.exists()]
            if missing:
                print(f"Skipping — files not found: {missing}")
                continue

            if cached_runs and r < len(cached_runs) and i < len(cached_runs[r]):
                predicted = cached_runs[r][i]
            else:
                print(f"Running: {entry.get('question', '')[:70]}...")
                predicted = await _predict_entry(entry, paper_paths, args.baseline)

            run_preds.append(predicted)
            metrics = _evaluate(predicted, entry["conflicts"])
            _print_results(metrics, entry["paper_files"])
            conf = _confusion(predicted, entry["conflicts"])
            _print_confusion(conf)
            # accumulate confusion across runs
            for gt_type, preds in conf.items():
                agg_conf.setdefault(gt_type, {})
                for k, v in preds.items():
                    agg_conf[gt_type][k] = agg_conf[gt_type].get(k, 0) + v
            sc, st = _specificity(predicted, entry["conflicts"])
            spec_correct += sc
            spec_total += st
            per_entry_overall.append(metrics["OVERALL"])

        saved_runs.append(run_preds)
        if per_entry_overall:
            runs_overall_f1.append(sum(m["f1"] for m in per_entry_overall) / len(per_entry_overall))
            runs_overall_p.append(sum(m["precision"] for m in per_entry_overall) / len(per_entry_overall))
            runs_overall_r.append(sum(m["recall"] for m in per_entry_overall) / len(per_entry_overall))

    if args.save_preds and preds_path:
        # Save flat single-run list if 1 run (legacy-compatible), else list-of-runs.
        to_save = saved_runs[0] if len(saved_runs) == 1 else saved_runs
        with open(preds_path, "w") as f:
            json.dump(to_save, f, indent=2)
        print(f"\nSaved predictions to {preds_path}")

    # Summary
    print(f"\n{'='*42}")
    print(f"SUMMARY ({n_runs} run(s), {'baseline' if args.baseline else 'three-stage'})")
    if runs_overall_f1:
        def ms(xs: list[float]) -> str:
            if len(xs) == 1:
                return f"{xs[0]:.2f}"
            return f"{statistics.mean(xs):.2f} +/- {statistics.pstdev(xs):.2f}"
        print(f"  Overall Precision: {ms(runs_overall_p)}")
        print(f"  Overall Recall:    {ms(runs_overall_r)}")
        print(f"  Overall F1:        {ms(runs_overall_f1)}")
        if len(runs_overall_f1) > 1:
            print(f"  F1 range:          [{min(runs_overall_f1):.2f}, {max(runs_overall_f1):.2f}]")
    if agg_conf:
        print("\n  Aggregate confusion (gt-type -> predicted label, summed over runs):")
        for gt_type, preds in agg_conf.items():
            parts = ", ".join(f"{k}={v}" for k, v in sorted(preds.items()))
            print(f"    {gt_type:<16} -> {parts}")
    if spec_total:
        print(f"\n  Specificity (negative controls left unflagged): {spec_correct}/{spec_total} = {spec_correct/spec_total:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate conflict detector precision/recall")
    parser.add_argument("--ground-truth", required=True, help="Path to JSONL ground truth file")
    parser.add_argument("--papers-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--preds-file", default=None, help="JSON file to save/load pipeline predictions")
    parser.add_argument("--save-preds", action="store_true", help="Run pipeline and save predictions to --preds-file")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs (variance). Ignored when loading cached preds.")
    parser.add_argument("--baseline", action="store_true", help="Use the monolithic single-prompt baseline instead of the three-stage pipeline")
    asyncio.run(main(parser.parse_args()))

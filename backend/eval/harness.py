#!/usr/bin/env python3
"""
Evaluation harness for the claim conflict detector.

Usage:
  python -m eval.harness --ground-truth eval/ground_truth/set1.jsonl --papers-dir /path/to/papers

Ground truth JSONL format (one JSON object per line):
  {
    "paper_files": ["paper1.pdf", "paper2.pdf"],
    "question": "...",
    "conflicts": [
      {"claim_a_text": "...", "claim_b_text": "...", "type": "CONTRADICT"}
    ]
  }

Outputs: precision, recall, F1 per relationship type and overall.
"""
from __future__ import annotations
import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import detect_conflicts, extract_claims, normalize_terminology


def _load_pdf_text(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        pages = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append(f"[Page {i + 1}]\n{text}")
        return "\n\n".join(pages)


async def _run_one(paper_files: list[Path], question: str) -> list[dict]:
    texts = [_load_pdf_text(p) for p in paper_files]
    titles = [p.stem for p in paper_files]

    all_claims_nested = await asyncio.gather(
        *[extract_claims(t, title, idx) for idx, (t, title) in enumerate(zip(texts, titles))]
    )
    claims = [c for nested in all_claims_nested for c in nested]
    term_conflicts = await normalize_terminology(claims)
    pairs = await detect_conflicts(claims, term_conflicts)
    return [p.model_dump() for p in pairs]


def _match(pred_text: str, truth_text: str, threshold: float = 0.4) -> bool:
    """Simple token overlap match."""
    p_tokens = set(pred_text.lower().split())
    t_tokens = set(truth_text.lower().split())
    if not t_tokens:
        return False
    return len(p_tokens & t_tokens) / len(t_tokens) >= threshold


def _evaluate(predicted_pairs: list[dict], ground_truth: list[dict]) -> dict:
    results: dict[str, dict] = {}
    all_tp = all_fp = all_fn = 0

    rel_types = ["CONTRADICT", "SUPPORT", "INCOMMENSURABLE"]
    for rel in rel_types:
        preds = [p for p in predicted_pairs if p["relationship"] == rel]
        truths = [t for t in ground_truth if t["type"] == rel]

        matched_truth = set()
        tp = 0
        for pred in preds:
            for i, truth in enumerate(truths):
                if i in matched_truth:
                    continue
                if _match(pred["claim_a_text"], truth["claim_a_text"]) and \
                   _match(pred["claim_b_text"], truth["claim_b_text"]):
                    tp += 1
                    matched_truth.add(i)
                    break
                if _match(pred["claim_a_text"], truth["claim_b_text"]) and \
                   _match(pred["claim_b_text"], truth["claim_a_text"]):
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


def _print_results(results: dict, paper_files: list[str]) -> None:
    print(f"\nPapers: {', '.join(paper_files)}")
    print(f"{'Type':<20} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 42)
    for rel, metrics in results.items():
        p = metrics["precision"]
        r = metrics["recall"]
        f = metrics["f1"]
        print(f"{rel:<20} {p:>6.2f} {r:>6.2f} {f:>6.2f}")


async def main(args: argparse.Namespace) -> None:
    gt_path = Path(args.ground_truth)
    papers_dir = Path(args.papers_dir)

    if not gt_path.exists():
        print(f"Ground truth file not found: {gt_path}")
        sys.exit(1)

    all_results: list[dict] = []
    with open(gt_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            paper_paths = [papers_dir / fn for fn in entry["paper_files"]]
            missing = [str(p) for p in paper_paths if not p.exists()]
            if missing:
                print(f"Skipping — files not found: {missing}")
                continue

            print(f"\nRunning: {entry.get('question', '')[:80]}...")
            predicted = await _run_one(paper_paths, entry["question"])
            metrics = _evaluate(predicted, entry["conflicts"])
            _print_results(metrics, entry["paper_files"])
            all_results.append(metrics)

    if all_results:
        avg_f1 = sum(r["OVERALL"]["f1"] for r in all_results) / len(all_results)
        avg_p = sum(r["OVERALL"]["precision"] for r in all_results) / len(all_results)
        avg_r = sum(r["OVERALL"]["recall"] for r in all_results) / len(all_results)
        print(f"\n{'='*42}")
        print(f"AGGREGATE ({len(all_results)} runs)")
        print(f"  Precision: {avg_p:.2f}  Recall: {avg_r:.2f}  F1: {avg_f1:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate conflict detector precision/recall")
    parser.add_argument("--ground-truth", required=True, help="Path to JSONL ground truth file")
    parser.add_argument("--papers-dir", required=True, help="Directory containing PDF files")
    asyncio.run(main(parser.parse_args()))

"""Print all predicted pairs for a paper pair — helps debug why eval scores 0."""
import asyncio
import sys
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import detect_conflicts, extract_claims, normalize_terminology


def _load_pdf(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n\n".join(
            f"[Page {i+1}]\n{page.extract_text() or ''}"
            for i, page in enumerate(pdf.pages)
        )


async def main(pdf_a: Path, pdf_b: Path, question: str) -> None:
    texts = [_load_pdf(pdf_a), _load_pdf(pdf_b)]
    titles = [pdf_a.stem, pdf_b.stem]

    print("Extracting claims...")
    all_claims = await asyncio.gather(
        *[extract_claims(t, title, idx) for idx, (t, title) in enumerate(zip(texts, titles))]
    )
    claims = [c for nested in all_claims for c in nested]
    print(f"  {len(claims)} claims extracted ({sum(1 for c in claims if c.paper_index==0)} + {sum(1 for c in claims if c.paper_index==1)})\n")

    print("Normalizing...")
    term_conflicts = await normalize_terminology(claims)

    print("Detecting conflicts...")
    pairs = await detect_conflicts(claims, term_conflicts)

    print(f"\n{len(pairs)} pairs detected:\n")
    for p in pairs:
        print(f"  [{p.relationship}]")
        print(f"    A ({p.paper_a_title[:30]}): {p.claim_a_text[:100]}")
        print(f"    B ({p.paper_b_title[:30]}): {p.claim_b_text[:100]}")
        print()


if __name__ == "__main__":
    papers_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_papers")
    pair = sys.argv[2] if len(sys.argv) > 2 else "cf_attn"

    if pair == "cf_attn":
        asyncio.run(main(
            papers_dir / "catastrophic_forgetting_ml.pdf",
            papers_dir / "attention_is_all_you_need.pdf",
            "How do neural networks learn and retain information?"
        ))
    elif pair == "cf_neuro":
        asyncio.run(main(
            papers_dir / "catastrophic_forgetting_ml.pdf",
            papers_dir / "attention_schema_neuroscience.pdf",
            "How do biological and artificial neural networks handle learning and attention?"
        ))
    elif pair == "attn_neuro":
        asyncio.run(main(
            papers_dir / "attention_is_all_you_need.pdf",
            papers_dir / "attention_schema_neuroscience.pdf",
            "What is attention and how does it work?"
        ))

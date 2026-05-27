"""
Stage 3: conflict detection across claims.
Input:  all claims + terminology glossary
Output: list[ClaimPair]
"""
from __future__ import annotations
import json
from .models import Claim, TermConflict, ClaimPair
from .llm_client import chat

SYSTEM_PROMPT = """You are a scientific conflict analyst. Your job is to identify meaningful relationships between claims from different papers.

For each pair of claims from DIFFERENT papers, classify:
- SUPPORT: the claims mutually reinforce each other and are consistent under the same conditions
- CONTRADICT: the claims cannot both be true — a genuine logical or empirical conflict
- INCOMMENSURABLE: the claims appear to conflict but are actually answering different questions, operating at different scopes, using different methods, or suffering from terminology drift (defined in the glossary)

Important rules:
- Only return pairs that are meaningful — skip pairs that simply address unrelated topics.
- For INCOMMENSURABLE, always explain WHY they can't be directly compared.
- For CONTRADICT, be conservative — only flag genuine contradictions, not mere emphasis differences.
- Do not compare claims from the same paper.
- Aim for 5-20 pairs total. Quality over quantity.

Return a JSON object with this exact schema:
{
  "pairs": [
    {
      "claim_a_id": "<id>",
      "claim_b_id": "<id>",
      "relationship": "SUPPORT" | "CONTRADICT" | "INCOMMENSURABLE",
      "explanation": "<one to two sentence explanation>",
      "terminology_note": "<optional: explain terminology drift if INCOMMENSURABLE>"
    }
  ]
}"""


async def detect_conflicts(
    claims: list[Claim],
    term_conflicts: list[TermConflict],
) -> list[ClaimPair]:
    if len(claims) < 2:
        return []

    payload = _build_payload(claims, term_conflicts)
    raw = await chat(
        messages=[{"role": "user", "content": payload}],
        system=SYSTEM_PROMPT,
        json_mode=True,
        temperature=0.0,
    )

    data = _parse_response(raw)
    claim_map = {c.id: c for c in claims}
    pairs: list[ClaimPair] = []
    for item in data.get("pairs", []):
        a = claim_map.get(item.get("claim_a_id", ""))
        b = claim_map.get(item.get("claim_b_id", ""))
        if not a or not b:
            continue
        pairs.append(
            ClaimPair(
                claim_a_id=a.id,
                claim_b_id=b.id,
                claim_a_text=a.claim_text,
                claim_b_text=b.claim_text,
                paper_a_title=a.paper_title,
                paper_b_title=b.paper_title,
                relationship=item.get("relationship", "INCOMMENSURABLE"),
                explanation=item.get("explanation", ""),
                terminology_note=item.get("terminology_note"),
            )
        )
    return pairs


def _build_payload(claims: list[Claim], term_conflicts: list[TermConflict]) -> str:
    parts = ["CLAIMS (grouped by paper):\n"]
    by_paper: dict[int, list[Claim]] = {}
    for c in claims:
        by_paper.setdefault(c.paper_index, []).append(c)

    for idx, paper_claims in sorted(by_paper.items()):
        title = paper_claims[0].paper_title
        parts.append(f"--- Paper {idx}: {title} ---")
        for c in paper_claims:
            parts.append(f"  [{c.id}] {c.claim_text}")
        parts.append("")

    if term_conflicts:
        parts.append("\nTERMINOLOGY GLOSSARY (divergences that may explain apparent conflicts):\n")
        for tc in term_conflicts:
            if tc.conflict_type == "DIVERGENT":
                parts.append(f"  Term '{tc.term}': {tc.explanation}")
                for d in tc.definitions:
                    parts.append(f"    - Paper {d.paper_index} ({d.paper_title}): {d.definition}")
                parts.append("")

    return "\n".join(parts)


def _parse_response(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.rfind("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
        raise ValueError(f"Cannot parse JSON from response: {raw[:300]}")

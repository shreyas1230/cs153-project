"""
Stage 2: cross-paper terminology normalization.
Input:  all claims from all papers
Output: list[TermConflict]
"""
from __future__ import annotations
import json
from .models import Claim, TermConflict, TermDefinition
from .llm_client import chat

SYSTEM_PROMPT = """You are a terminology analyst comparing how multiple scientific papers use technical terms.

Given a set of claims from different papers, each tagged with key_terms:
1. Identify terms that appear in multiple papers.
2. For each shared term, determine how each paper uses or defines it.
3. Classify the relationship:
   - DIVERGENT: same word, meaningfully different definitions or referents (this is the most interesting case)
   - RELATED: different words for the same concept
   - CONSISTENT: same term, same meaning across papers

Only include terms where there is something worth noting. Skip trivial shared vocabulary.

Return a JSON object with this exact schema:
{
  "glossary": [
    {
      "term": "<term>",
      "conflict_type": "DIVERGENT" | "RELATED" | "CONSISTENT",
      "explanation": "<one sentence explaining the conflict or relationship>",
      "definitions": [
        {
          "paper_index": <int>,
          "paper_title": "<string>",
          "definition": "<how this paper uses the term>"
        }
      ]
    }
  ]
}"""


async def normalize_terminology(claims: list[Claim]) -> list[TermConflict]:
    # Question-BLIND by design: the glossary should cover all cross-paper terminology
    # drift, not only terms the question mentions. See docs/question-steering.md.
    if not claims:
        return []

    claims_summary = _build_claims_summary(claims)
    raw = await chat(
        messages=[{"role": "user", "content": claims_summary}],
        system=SYSTEM_PROMPT,
        json_mode=True,
        temperature=0.0,
    )

    data = _parse_response(raw)
    conflicts: list[TermConflict] = []
    for item in data.get("glossary", []):
        defs = [
            TermDefinition(
                term=item["term"],
                paper_index=d["paper_index"],
                paper_title=d.get("paper_title", ""),
                definition=d["definition"],
            )
            for d in item.get("definitions", [])
        ]
        conflicts.append(
            TermConflict(
                term=item["term"],
                definitions=defs,
                conflict_type=item.get("conflict_type", "CONSISTENT"),
                explanation=item.get("explanation", ""),
            )
        )
    return conflicts


def _build_claims_summary(claims: list[Claim]) -> str:
    lines = ["Claims extracted from papers:\n"]
    for c in claims:
        terms = ", ".join(c.key_terms) if c.key_terms else "none"
        lines.append(
            f"[Paper {c.paper_index}: {c.paper_title}]\n"
            f"  Claim: {c.claim_text}\n"
            f"  Key terms: {terms}\n"
        )
    return "\n".join(lines)


def _parse_response(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.rfind("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
        raise ValueError(f"Cannot parse JSON from response: {raw[:300]}")

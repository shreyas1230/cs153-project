"""
Stage 1: extract atomic claims from a single paper's text.
Input:  paper text (string), paper metadata
Output: list[Claim]
"""
from __future__ import annotations
import json
from .models import Claim
from .llm_client import chat

SYSTEM_PROMPT = """You are a scientific claims extractor. Your job is to read a scientific paper and extract every atomic, falsifiable claim it makes.

Rules:
- Each claim must be a single, specific assertion (not a general topic or method description).
- Each claim must be grounded in evidence or data presented in the paper.
- Copy the source_passage verbatim — do NOT paraphrase.
- Tag key_terms: the specific technical terms this claim depends on.
- Tag domain_signals: the research fields this paper operates in (e.g. "machine learning", "cognitive neuroscience").
- Aim for 10-30 claims per paper. Skip pure methodology descriptions; focus on results and assertions.

Return a JSON object with this exact schema:
{
  "paper_title": "<string>",
  "claims": [
    {
      "claim_text": "<one sentence assertion>",
      "source_passage": "<verbatim excerpt>",
      "page_number": <int or null>,
      "key_terms": ["<term1>", "<term2>"],
      "domain_signals": ["<field1>"]
    }
  ]
}"""


async def extract_claims(
    text: str,
    paper_title: str,
    paper_index: int,
) -> list[Claim]:
    user_msg = (
        f"Paper title: {paper_title}\n\n"
        f"Full text:\n{text[:50000]}"  # guard against extreme length
    )
    raw = await chat(
        messages=[{"role": "user", "content": user_msg}],
        system=SYSTEM_PROMPT,
        json_mode=True,
        temperature=0.0,
    )

    data = _parse_response(raw)
    claims: list[Claim] = []
    for item in data.get("claims", []):
        claims.append(
            Claim(
                paper_title=paper_title,
                paper_index=paper_index,
                claim_text=item.get("claim_text", ""),
                source_passage=item.get("source_passage", ""),
                page_number=item.get("page_number"),
                key_terms=item.get("key_terms", []),
                domain_signals=item.get("domain_signals", []),
            )
        )
    return claims


def _parse_response(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.rfind("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
        raise ValueError(f"Cannot parse JSON from response: {raw[:300]}")

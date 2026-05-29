from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field
import uuid


class Claim(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    paper_title: str
    paper_index: int
    claim_text: str
    source_passage: str
    page_number: Optional[int] = None
    key_terms: list[str] = Field(default_factory=list)
    domain_signals: list[str] = Field(default_factory=list)


class TermDefinition(BaseModel):
    term: str
    paper_index: int
    paper_title: str
    definition: str


class TermConflict(BaseModel):
    term: str
    definitions: list[TermDefinition]
    conflict_type: Literal["CONSISTENT", "RELATED", "DIVERGENT"]
    explanation: str


class ClaimPair(BaseModel):
    claim_a_id: str
    claim_b_id: str
    claim_a_text: str
    claim_b_text: str
    paper_a_title: str
    paper_b_title: str
    relationship: Literal["SUPPORT", "CONTRADICT", "INCOMMENSURABLE"]
    explanation: str
    terminology_note: Optional[str] = None


class UsageSummary(BaseModel):
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    llm_calls: int = 0


class AnalysisResult(BaseModel):
    session_id: str
    question: str
    papers: list[str]
    claims: list[Claim]
    term_conflicts: list[TermConflict]
    claim_pairs: list[ClaimPair]
    status: Literal["extracting", "normalizing", "detecting", "done", "error"] = "extracting"
    error: Optional[str] = None
    usage: UsageSummary = Field(default_factory=UsageSummary)

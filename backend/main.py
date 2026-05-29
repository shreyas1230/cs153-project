from __future__ import annotations
import asyncio
import io
import json
import os
import uuid
from pathlib import Path
from typing import Optional

import pdfplumber
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline import detect_conflicts, extract_claims, normalize_terminology
from pipeline.models import AnalysisResult, UsageSummary
from pipeline.llm_client import set_usage_tracker, reset_usage_tracker

app = FastAPI(title="Claim Conflict Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


# ---------- helpers ----------

def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def _save(result: AnalysisResult) -> None:
    _session_path(result.session_id).write_text(result.model_dump_json(indent=2))


def _load(session_id: str) -> Optional[AnalysisResult]:
    p = _session_path(session_id)
    if not p.exists():
        return None
    return AnalysisResult.model_validate_json(p.read_text())


def _extract_text(pdf_bytes: bytes) -> tuple[str, int | None]:
    """Returns (full_text, page_count). Falls back to empty string on failure."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append(f"[Page {i + 1}]\n{text}")
            return "\n\n".join(pages), len(pdf.pages)
    except Exception:
        return "", None


# ---------- background pipeline ----------

async def _run_pipeline(session_id: str, texts: list[str], titles: list[str], question: str) -> None:
    result = _load(session_id)
    if result is None:
        return

    usage_records: list = []
    token = set_usage_tracker(usage_records)
    try:
        # Stage 1: extract claims from each paper in parallel
        result.status = "extracting"
        _save(result)

        tasks = [
            extract_claims(text, title, idx)
            for idx, (text, title) in enumerate(zip(texts, titles))
        ]
        all_claims_nested = await asyncio.gather(*tasks)
        result.claims = [c for nested in all_claims_nested for c in nested]

        # Stage 2: normalize terminology
        result.status = "normalizing"
        _save(result)
        result.term_conflicts = await normalize_terminology(result.claims)

        # Stage 3: detect conflicts
        result.status = "detecting"
        _save(result)
        result.claim_pairs = await detect_conflicts(result.claims, result.term_conflicts)

        result.usage = UsageSummary(
            total_tokens=sum(u["tokens"] for u in usage_records),
            total_cost_usd=sum(u["cost"] for u in usage_records),
            llm_calls=len(usage_records),
        )
        result.status = "done"
        _save(result)

    except Exception as exc:
        result.status = "error"
        result.error = str(exc)
        _save(result)
    finally:
        reset_usage_tracker(token)


# ---------- endpoints ----------

@app.post("/api/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    question: str = Form(...),
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF is required.")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 PDFs per analysis.")

    session_id = str(uuid.uuid4())
    texts: list[str] = []
    titles: list[str] = []

    for f in files:
        raw = await f.read()
        text, _ = _extract_text(raw)
        if not text.strip():
            raise HTTPException(
                status_code=422,
                detail=f"Could not extract text from '{f.filename}'. It may be a scanned PDF.",
            )
        texts.append(text)
        titles.append(Path(f.filename or "Unknown").stem)

    result = AnalysisResult(
        session_id=session_id,
        question=question,
        papers=titles,
        claims=[],
        term_conflicts=[],
        claim_pairs=[],
        status="extracting",
    )
    _save(result)

    background_tasks.add_task(_run_pipeline, session_id, texts, titles, question)
    return {"session_id": session_id}


@app.get("/api/status/{session_id}")
async def status(session_id: str):
    result = _load(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    stage_progress = {
        "extracting": 10,
        "normalizing": 50,
        "detecting": 75,
        "done": 100,
        "error": 0,
    }
    return {
        "status": result.status,
        "progress": stage_progress.get(result.status, 0),
        "error": result.error,
    }


@app.get("/api/results/{session_id}")
async def results(session_id: str):
    result = _load(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if result.status not in ("done", "error"):
        raise HTTPException(status_code=202, detail="Analysis still in progress.")
    return result.model_dump()


@app.get("/health")
async def health():
    return {"ok": True}

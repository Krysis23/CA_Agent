"""
FastAPI backend:
- /upload: Gemini vision extraction (structured or plain text)
- /ask: User question + uploaded file context (structured + plain extracts) are combined for
  RAG and for the LLM; recent chat turns are appended separately for memory.
  Metrics (retrieval eval + answer judge) are optional — toggled via dev_config.py.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from search import search_rag
from ca_agent import process_query
from file_handler import extract_bank_summary_with_gemini
from dev_config import ENABLE_METRICS
from metrics import evaluate_retrieval, judge_answer   # both return None when ENABLE_METRICS = False

safe = lambda x: float(x) if isinstance(x, (int, float)) else 0

app = FastAPI(title="CA AI Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_DOCUMENT_CONTEXT_CHARS = 3200
MAX_COMBINED_QUERY_CHARS   = 4000
MAX_FINAL_QUERY_CHARS      = 5500


def _as_float(val: Any) -> float:
    try:
        return float(val if val is not None else 0)
    except (TypeError, ValueError):
        return 0.0


def _nonzero_amount(val: Any) -> bool:
    return abs(_as_float(val)) > 1e-9


def sanitize_tax_breakdown(tb: Any) -> dict[str, Any] | None:
    """Keep numeric tax_breakdown from Gemini JSON (JSON-serializable, bounded list)."""
    if not isinstance(tb, dict):
        return None
    out: dict[str, Any] = {}
    for k in (
        "gross_income",
        "standard_deduction",
        "taxable_income",
        "rebate_87a",
        "base_tax",
        "cess_4_percent",
        "final_tax",
    ):
        if k in tb:
            out[k] = _as_float(tb[k])
    slabs = tb.get("slab_breakdown")
    if isinstance(slabs, list):
        rows: list[dict[str, Any]] = []
        for s in slabs[:24]:
            if not isinstance(s, dict):
                continue
            rows.append({
                "range": str(s.get("range", "") or ""),
                "rate": str(s.get("rate", "") or ""),
                "tax": _as_float(s.get("tax", 0)),
            })
        if rows:
            out["slab_breakdown"] = rows
    return out if out else None


def slim_chat_context_for_storage(cc: Any) -> dict[str, Any] | None:
    """Subset of Gemini chat_context for persistence (avoid huge blobs in localStorage)."""
    if not isinstance(cc, dict):
        return None
    out: dict[str, Any] = {}
    dt = cc.get("document_type")
    if dt:
        out["document_type"] = str(dt)
    bs = cc.get("bank_summary")
    if isinstance(bs, dict):
        slim_bs: dict[str, float] = {}
        for k in ("opening_balance", "credits", "debits", "closing_balance"):
            if k in bs and bs[k] is not None:
                slim_bs[k] = _as_float(bs[k])
        if slim_bs:
            out["bank_summary"] = slim_bs
    tc = cc.get("tax_context")
    if isinstance(tc, dict):
        slim_tc: dict[str, Any] = {}
        for k in ("regime", "fy", "estimated_tax", "taxable_income", "standard_deduction", "rebate_87a"):
            if k not in tc or tc[k] is None:
                continue
            v = tc[k]
            if isinstance(v, (int, float)):
                slim_tc[k] = float(v)
            else:
                slim_tc[k] = v
        if slim_tc:
            out["tax_context"] = slim_tc
    return out if out else None


def doc_data_from_bank_gemini(parsed: dict[str, Any]) -> dict[str, Any]:
    """Structured fields + tax_breakdown for chat persistence and /ask context."""
    narr = str(parsed.get("summary_text") or "").strip()
    tb = sanitize_tax_breakdown(parsed.get("tax_breakdown"))
    cc = slim_chat_context_for_storage(parsed.get("chat_context"))

    doc: dict[str, Any] = {
        "document_type": "bank_statement",
        "person_name": parsed.get("person_entity") or "N/A",
        "period": parsed.get("statement_period") or "N/A",
        "raw_text_summary": narr or "N/A",
        "bank": {
            "opening_balance": _as_float(parsed.get("opening_balance")),
            "total_credits": _as_float(parsed.get("total_credits")),
            "total_debits": _as_float(parsed.get("total_debits")),
            "closing_balance": _as_float(parsed.get("closing_balance")),
        },
        "estimated_annual_income": _as_float(parsed.get("estimated_annual_income")),
        "taxable_income_estimate": _as_float(parsed.get("taxable_income_estimate")),
        "estimated_tax_new_regime_fy_2025_26": _as_float(parsed.get("estimated_tax_new_regime_fy_2025_26")),
    }
    if tb:
        doc["tax_breakdown"] = tb
    if cc:
        doc["chat_context"] = cc
    return doc


def compact_doc_retrieval_hint(doc_data: Optional[dict[str, Any]]) -> str:
    """Compact line appended to the embedding text for RAG (not sent to HyDE prompt)."""
    if not doc_data:
        return ""
    bits: list[str] = []
    pn = str(doc_data.get("person_name") or "").strip()
    if pn and pn != "N/A":
        bits.append(f"Uploaded bank statement context for {pn}")
    est = _as_float(doc_data.get("estimated_annual_income"))
    if est > 0:
        bits.append(f"estimated annual income {est:.0f} INR")
    tb = doc_data.get("tax_breakdown")
    if isinstance(tb, dict):
        ti = _as_float(tb.get("taxable_income"))
        ft = _as_float(tb.get("final_tax"))
        if ti > 0:
            bits.append(f"model taxable income {ti:.0f} INR (FY new regime)")
        if ft > 0:
            bits.append(f"model final tax per slabs {ft:.0f} INR")
    et = _as_float(doc_data.get("estimated_tax_new_regime_fy_2025_26"))
    if et > 0 and not (isinstance(tb, dict) and _as_float(tb.get("final_tax")) > 0):
        bits.append(f"model estimated tax {et:.0f} INR")
    bank = doc_data.get("bank") or {}
    if isinstance(bank, dict):
        cr = _as_float(bank.get("total_credits"))
        if cr > 0:
            bits.append(f"total credits on statement {cr:.0f} INR")
    return ". ".join(bits)[:650]


class HistoryTurn(BaseModel):
    user: str
    assistant: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[HistoryTurn] = Field(default_factory=list)
    plain_file_texts: list[str] = Field(default_factory=list)
    doc_data: Optional[dict[str, Any]] = None


def build_memory(history: list[HistoryTurn]) -> str:
    """Last few chat turns only — file text lives in build_document_context."""
    return "\n".join(
        f"User: {t.user}\nAI: {t.assistant}" for t in history[-3:]
    ).strip()


def build_document_context(
    doc_data: Optional[dict[str, Any]],
    plain_file_texts: list[str],
) -> str:
    parts: list[str] = []
    if doc_data:
        parts.append(
            "[Uploaded document — structured]\n"
            + format_structured_doc_summary(doc_data)
        )
    if plain_file_texts:
        blob = "\n\n".join(t.strip() for t in plain_file_texts if t and str(t).strip())
        if blob:
            parts.append("[Uploaded document — extracted text]\n" + blob)
    joined = "\n\n".join(parts).strip()
    return joined[:MAX_DOCUMENT_CONTEXT_CHARS] if joined else ""


def combine_question_and_uploads(user_text: str, doc_blob: str) -> str:
    if not doc_blob:
        return user_text.strip()
    return (user_text.strip() + "\n\n" + doc_blob).strip()[:MAX_COMBINED_QUERY_CHARS]


def chunks_to_preview_records(chunks: pd.DataFrame, text_limit: int = 300) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _, row in chunks.iterrows():
        cid = row.get("chunk_id", 0)
        try:
            cid = int(cid)
        except (TypeError, ValueError):
            cid = 0
        txt = row.get("text", "") or ""
        out.append({
            "level":    str(row.get("level", "") or ""),
            "book":     str(row.get("book", "") or ""),
            "chunk_id": cid,
            "text":     txt[:text_limit],
        })
    return out


def format_structured_doc_summary(doc_data: dict[str, Any]) -> str:
    is_new = "person_entity" in doc_data
    is_api_bank = (
        not is_new
        and doc_data.get("document_type") == "bank_statement"
        and ("person_name" in doc_data or doc_data.get("bank"))
    )

    if is_new:
        person = doc_data.get("person_entity", "N/A")
        period = doc_data.get("statement_period", "N/A")
        chat = doc_data.get("chat_context") or {}
        bank = chat.get("bank_summary") or {}
        inc = chat.get("income") or {}
        tb = doc_data.get("tax_breakdown") or {}
        credits = safe(doc_data.get("total_credits", 0))
        debits = safe(doc_data.get("total_debits", 0))
        opening = safe(doc_data.get("opening_balance", 0))
        business = safe(inc.get("business_income", 0)) or safe(doc_data.get("estimated_annual_income", 0))
        summary = str(doc_data.get("summary_text", "") or "")
        est_tax = safe(doc_data.get("estimated_tax_new_regime_fy_2025_26", 0))
        taxable = safe(doc_data.get("taxable_income_estimate", 0))
    elif is_api_bank:
        person = doc_data.get("person_name", "N/A")
        period = doc_data.get("period", "N/A")
        chat = doc_data.get("chat_context") or {}
        bank = chat.get("bank_summary") or {}
        bank_d = doc_data.get("bank") or {}
        inc = chat.get("income") or {}
        tb = doc_data.get("tax_breakdown") or {}
        credits = safe(bank_d.get("total_credits", 0)) or safe(bank.get("credits", 0))
        debits = safe(bank_d.get("total_debits", 0)) or safe(bank.get("debits", 0))
        opening = safe(bank_d.get("opening_balance", 0)) or safe(bank.get("opening_balance", 0))
        business = safe(inc.get("business_income", 0)) or safe(doc_data.get("estimated_annual_income", 0))
        summary = str(doc_data.get("raw_text_summary", "") or "")
        est_tax = safe(doc_data.get("estimated_tax_new_regime_fy_2025_26", 0))
        taxable = safe(doc_data.get("taxable_income_estimate", 0))
    else:
        person = doc_data.get("person_name", "N/A")
        period = doc_data.get("period", "N/A")
        bank_d = doc_data.get("bank") or {}
        inc = doc_data.get("income") or {}
        tb = doc_data.get("tax_breakdown") or {}
        credits = safe(bank_d.get("total_credits", 0))
        debits = safe(bank_d.get("total_debits", 0))
        opening = safe(bank_d.get("opening_balance", 0))
        business = safe(inc.get("business_income", 0)) or safe(doc_data.get("estimated_annual_income", 0))
        summary = str(doc_data.get("raw_text_summary", "") or "")
        est_tax = safe(doc_data.get("estimated_tax_new_regime_fy_2025_26", 0))
        taxable = safe(doc_data.get("taxable_income_estimate", 0))

    lines = [
        "### Bank Statement",
        f"- **Person/Entity:** {person}",
        f"- **Period:** {period}",
        f"- **Summary:** {summary}",
        "",
        "**Bank Summary**",
        f"- Opening balance: \u20b9{safe(opening):,.2f}",
        f"- Total credits: \u20b9{safe(credits):,.2f}",
        f"- Total debits: \u20b9{safe(debits):,.2f}",
        "",
        "**Estimated Income & Tax (New Regime FY 2025-26)**",
        f"- Estimated annual income: \u20b9{safe(business):,.2f}",
        f"- Taxable income: \u20b9{safe(taxable):,.2f}",
        f"- Estimated tax: \u20b9{safe(est_tax):,.2f}",
    ]

    slabs = tb.get("slab_breakdown", []) if isinstance(tb, dict) else []
    active = [s for s in slabs if isinstance(s, dict) and safe(s.get("tax", 0)) > 0]
    if active:
        lines.append("- Slab breakdown:")
        for s in active:
            lines.append(f"  - {s.get('range')} @ {s.get('rate')} = \u20b9{safe(s.get('tax', 0)):,.2f}")

    return "\n".join(lines).strip()


def bank_gemini_dict_to_plaintext(parsed: dict[str, Any]) -> str:
    """Lines matching frontend parseBankSummary() labels (Person/Entity:, Opening balance:, …)."""
    def money(val: Any) -> str:
        try:
            return f"₹{float(val):,.2f}"
        except (TypeError, ValueError):
            return "N/A"

    pe = str(parsed.get("person_entity") or "").strip() or "N/A"
    period = str(parsed.get("statement_period") or "").strip() or "N/A"
    return "\n".join(
        [
            f"Person/Entity: {pe}",
            f"Period: {period}",
            f"Opening balance: {money(parsed.get('opening_balance', 0))}",
            f"Total credits: {money(parsed.get('total_credits', 0))}",
            f"Total debits: {money(parsed.get('total_debits', 0))}",
            f"Closing balance: {money(parsed.get('closing_balance', 0))}",
            f"Estimated annual income: {money(parsed.get('estimated_annual_income', 0))}",
            f"Estimated tax: {money(parsed.get('estimated_tax_new_regime_fy_2025_26', 0))}",
        ]
    )


@app.post("/ask")
async def ask(body: AskRequest):
    user_text = body.question.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty question")

    doc_blob       = build_document_context(body.doc_data, body.plain_file_texts)
    combined_query = combine_question_and_uploads(user_text, doc_blob)
    memory_context = build_memory(body.history)
    final_query    = (combined_query + ("\n\n" + memory_context if memory_context else "")).strip()
    final_query    = final_query[:MAX_FINAL_QUERY_CHARS]

    try:
        # RAG: question + optional upload hint; HyDE returns classification in all modes.
        rag_hint = compact_doc_retrieval_hint(body.doc_data)
        chunks, classification = search_rag(user_text, top_k=8, doc_retrieval_hint=rag_hint or None)

        level_order = {"final": 0, "intermediate": 1, "foundation": 2}
        if "level" in chunks.columns:
            chunks = chunks.copy()
            chunks["level_rank"] = chunks["level"].map(level_order)
            chunks = chunks.sort_values(
                by=["level_rank", "rerank_score"],
                ascending=[True, False],
            )

        retrieved_context = chunks.to_json(orient="records")
        answer = process_query(
            final_query,
            retrieved_context,
            doc_data=body.doc_data,
            classification=classification,
        )

        # ── Metrics — only runs when ENABLE_METRICS = True ────────
        metrics = None
        if ENABLE_METRICS:
            retrieval = evaluate_retrieval(user_text, chunks, k=4)
            judge     = judge_answer(user_text, answer)

            if retrieval and judge:
                metrics = {
                    "precision_at_4": retrieval["precision_at_k"],
                    "crag_status":    retrieval["crag_status"],
                    "crag_score":     retrieval["crag_score"],
                    "crag_action":    retrieval["crag_action"],
                    "chunk_scores":   retrieval["chunk_scores"],
                    "judge_score":    judge["score"],
                    "judge_reason":   judge["reason"],
                }

        if metrics:
            print("\n" + "=" * 60)
            print("  METRICS REPORT")
            print("=" * 60)
            print(f"  Precision@4:   {metrics['precision_at_4']}")
            print(f"  CRAG Status:   {metrics['crag_status']}")
            print(f"  CRAG Score:    {metrics['crag_score']}")
            print(f"  CRAG Action:   {metrics['crag_action']}")
            print(f"  Chunk Scores:  {metrics['chunk_scores']}")
            print(f"  Judge Score:   {metrics['judge_score']} / 5")
            print(f"  Judge Reason:  {metrics['judge_reason']}")
            print("=" * 60 + "\n")

        return {
            "answer":           answer,
            "metrics":          metrics,          # null in prod, populated in dev
            "retrieved_chunks": chunks_to_preview_records(chunks),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower() or ".bin"
    safe_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        display_name = Path(file.filename or "document").name

        ext_l = ext.lstrip(".").lower()
        if ext_l not in ("pdf", "png", "jpg", "jpeg", "webp"):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Use pdf, png, jpg, jpeg, or webp.",
            )

        parsed = extract_bank_summary_with_gemini(file_path)

        if not parsed or not isinstance(parsed, dict):
            raise HTTPException(
                status_code=422,
                detail=f"Could not read any content from {display_name}.",
            )

        plaintext = bank_gemini_dict_to_plaintext(parsed)
        narrative = str(parsed.get("summary_text") or "").strip()
        summary_message = plaintext + (f"\n\n{narrative}" if narrative else "")

        doc_data = doc_data_from_bank_gemini(parsed)

        return {
            "doc_item":         {"filename": display_name, "summary": summary_message},
            "summary_message":  summary_message,
            "doc_data":         doc_data,
            "plain_text_append": None,
            "warning":          None,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass
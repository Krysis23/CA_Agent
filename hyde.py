"""
hyde.py
─────────────────────────────────────────────────────────────────────
HyDE + Query Classification

ENABLE_HYDE = True:
  One Gemini call (DEV_GEMINI_API_KEY) returns:
    1. hypothetical_document  — ICAI-style paragraph for embedding blend
    2. classification         — query routing metadata

ENABLE_HYDE = False:
  Zero Gemini calls. Python keyword matching returns classification only.
  hypothetical_document is None. Zero cost. Zero quota.

Return contract (both modes):
  {
    "hypothetical_document": str | None,
    "classification": {
      "query_type":        "theory" | "numerical" | "theory+numerical" | "document",
      "tax_type":          "income_tax" | "gst" | "none",
      "needs_calculation": bool,
      "regime":            "old" | "new"
    }
  }

Only imported by search.py.
"""

from __future__ import annotations

import json
import re

from dev_config import ENABLE_HYDE, DEV_GEMINI_API_KEY, DEV_MODEL

_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    if not DEV_GEMINI_API_KEY:
        raise EnvironmentError(
            "[HyDE] DEV_GEMINI_API_KEY not set. "
            "Add it to .env or set ENABLE_HYDE=false."
        )
    import google.generativeai as genai

    genai.configure(api_key=DEV_GEMINI_API_KEY)
    _model = genai.GenerativeModel(DEV_MODEL)
    return _model


# ── Keyword fallback (used ONLY when ENABLE_HYDE = False) ─────────


def _kw_query_type(q: str) -> str:
    q = q.lower()
    has_calc = any(
        k in q
        for k in [
            "calculate",
            "compute",
            "work out",
            "solve",
            "how much",
            "liability",
            "tax payable",
            "net gst",
            "show working",
            "step by step",
            "numerical",
            "breakdown",
            "break down",
            "give me",
            "show me",
            "slab",
            "detail",
            "working",
            "tax calculation",
            "tax liability",
            "tax amount",
            "estimate",
        ]
    )
    has_theory = any(
        k in q
        for k in [
            "explain",
            "describe",
            "what is",
            "why",
            "how does",
            "difference",
            "define",
            "meaning",
            "concept",
        ]
    )
    if has_calc and has_theory:
        return "theory+numerical"
    if has_calc:
        return "numerical"
    if has_theory:
        return "theory"
    return "theory"


def _kw_tax_type(q: str) -> str:
    q = q.lower()
    if any(k in q for k in ["gst", "cgst", "sgst", "igst", "input tax credit", "output tax", "rcm"]):
        return "gst"
    if any(
        k in q
        for k in [
            "tax",
            "80c",
            "80d",
            "income",
            "salary",
            "tds",
            "capital gains",
            "regime",
            "deduction",
        ]
    ):
        return "income_tax"
    return "none"


def _kw_needs_calculation(q: str) -> bool:
    q = q.lower()
    return any(
        k in q
        for k in [
            "calculate",
            "compute",
            "work out",
            "solve",
            "how much",
            "liability",
            "tax payable",
            "net gst",
            "show working",
            "step by step",
            "numerical",
            "breakdown",
            "break down",
            "give me",
            "show me",
            "slab",
            "detail",
            "working",
            "tax calculation",
            "tax liability",
            "tax amount",
            "estimate",
        ]
    )


def _kw_regime(q: str) -> str:
    return "old" if "old regime" in q.lower() else "new"


def _keyword_classification(query: str) -> dict:
    """Pure Python classification — no API call, used when ENABLE_HYDE = False."""
    return {
        "query_type":        _kw_query_type(query),
        "tax_type":          _kw_tax_type(query),
        "needs_calculation": _kw_needs_calculation(query),
        "regime":            _kw_regime(query),
    }


# ── Gemini classification + HyDE (used ONLY when ENABLE_HYDE = True) ─

_GEMINI_PROMPT = """You are a CA expert, ICAI textbook author, and query classifier.

Given the user query below, return ONLY strict valid JSON — no markdown, no code fences, no explanation.

Query: {query}
{hint_block}

Return this exact JSON structure:

{{
  "hypothetical_document": "<100-word ICAI-style paragraph that directly answers the query, using formal language and section references where relevant>",
  "classification": {{
    "query_type": "<one of: theory | numerical | theory+numerical | document>",
    "tax_type": "<one of: income_tax | gst | none>",
    "needs_calculation": <true | false>,
    "regime": "<one of: old | new>"
  }}
}}

Classification rules:
- query_type:
    theory           = student wants concept/definition/rule explanation only
    numerical        = student wants a calculation only
    theory+numerical = student wants both explanation AND calculation
    document         = question is specifically about the uploaded document

- tax_type:
    income_tax = involves income tax, slabs, deductions, TDS, capital gains
    gst        = involves GST, CGST, SGST, IGST, input tax credit, output tax, RCM
    none       = neither

- needs_calculation:
    true  = any numbers need to be computed
    false = purely conceptual

- regime:
    old = query mentions old regime, or is a general tax question (default)
    new = query explicitly mentions new regime

hypothetical_document rules:
- Write exactly one paragraph (~100 words)
- Use formal ICAI textbook language
- Include section numbers where relevant
- Preserve all numerical values exactly
- Write ONLY the paragraph — no heading, no label, no extra text inside the string

CRITICAL: Return ONLY the JSON object. Nothing else."""


def _extract_json(text: str) -> str:
    """Extract the outermost JSON object from text."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Unmatched braces in Gemini response")


def _validate_classification(cls: dict, query: str) -> dict:
    """Ensure all classification fields are valid; fall back to keyword for any bad value."""
    valid_qt = {"theory", "numerical", "theory+numerical", "document"}
    valid_tax = {"income_tax", "gst", "none"}
    valid_regimes = {"old", "new"}

    qt = cls.get("query_type")
    tt = cls.get("tax_type")
    rg = cls.get("regime")
    nc = cls.get("needs_calculation")

    if isinstance(nc, bool):
        needs_calc = nc
    else:
        needs_calc = _kw_needs_calculation(query)

    return {
        "query_type":        qt if qt in valid_qt else _kw_query_type(query),
        "tax_type":          tt if tt in valid_tax else _kw_tax_type(query),
        "needs_calculation": needs_calc,
        "regime":            rg if rg in valid_regimes else _kw_regime(query),
    }


def hypothetical_document(
    query: str,
    context_hint: str | None = None,
) -> dict:
    """
    Main entry point called by search.py.

    ENABLE_HYDE = True:
      Calls Gemini once. Returns hypothetical_document + Gemini classification.
      Falls back to keyword classification if Gemini fails.

    ENABLE_HYDE = False:
      No API call. Returns hypothetical_document=None + keyword classification.

    Always returns a dict — never None — so callers can always read .classification.
    """

    if not ENABLE_HYDE:
        return {
            "hypothetical_document": None,
            "classification":        _keyword_classification(query),
        }

    hint_block = ""
    if context_hint and str(context_hint).strip():
        hint_block = (
            "\nAdditional factual context from uploaded document "
            "(use for the hypothetical paragraph; do not contradict):\n"
            + str(context_hint).strip()[:1200]
        )

    prompt = _GEMINI_PROMPT.format(
        query=query.strip(),
        hint_block=hint_block,
    )

    try:
        print("[GEMINI DEV CALL] hyde.hypothetical_document()")
        raw = _get_model().generate_content(prompt).text.strip()
        raw = re.sub(r"```json|```", "", raw, flags=re.IGNORECASE).strip()

        parsed = json.loads(_extract_json(raw))

        hyde_text = str(parsed.get("hypothetical_document", "")).strip()
        raw_cls = parsed.get("classification") or {}
        if not isinstance(raw_cls, dict):
            raw_cls = {}

        classification = _validate_classification(raw_cls, query)

        if not hyde_text:
            raise ValueError("Empty hypothetical_document in Gemini response")

        print(f"[HyDE] Generated: {hyde_text[:80]}...")
        print(f"[HyDE] Classification: {classification}")

        return {
            "hypothetical_document": hyde_text,
            "classification":        classification,
        }

    except Exception as e:
        print(f"[HyDE] Gemini call failed — using keyword classification. Error: {e}")
        return {
            "hypothetical_document": None,
            "classification":        _keyword_classification(query),
        }

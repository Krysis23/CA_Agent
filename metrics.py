"""
metrics.py
─────────────────────────────────────────────────────────────────────
Retrieval evaluation + answer judge — developer feature only.

ENABLE_METRICS = False  →  evaluate_retrieval() and judge_answer() both
                            return None immediately. No client created.
                            Zero cost. Zero quota used.

ENABLE_METRICS = True   →  Uses DEV_GEMINI_API_KEY exclusively.
                            Never touches GEMINI_API_KEY.

api.py checks for None before using results, so the response shape
is always valid — metrics key is just null in production.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import pandas as pd

from dev_config import ENABLE_METRICS, DEV_GEMINI_API_KEY, DEV_MODEL

_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model

    if not DEV_GEMINI_API_KEY:
        raise EnvironmentError(
            "[Metrics] DEV_GEMINI_API_KEY is not set in .env.\n"
            "Either add it or set ENABLE_METRICS = False in dev_config.py."
        )

    import google.generativeai as genai
    genai.configure(api_key=DEV_GEMINI_API_KEY)
    _model = genai.GenerativeModel(DEV_MODEL)
    return _model


# ═══════════════════════════════════════════════════════════════════
# INTERNAL — score all chunks in a single Gemini call
# ═══════════════════════════════════════════════════════════════════

def _get_response_text(response):
    if response is None:
        raise ValueError("Gemini response is None")

    if hasattr(response, "text"):
        try:
            return response.text
        except Exception as e:
            response_details = None
            if hasattr(response, "to_dict"):
                try:
                    response_details = json.dumps(response.to_dict(), default=str)
                except Exception:
                    response_details = "<unable to serialize response.to_dict()>"
            raise ValueError(
                f"Gemini response has no valid text parts.\nOriginal error: {e}\nResponse details: {response_details}"
            ) from e

    if hasattr(response, "candidates"):
        try:
            candidates = response.candidates
        except Exception:
            candidates = []
        if candidates:
            candidate = candidates[0]
            content = getattr(candidate, "content", None)
            if content is not None:
                parts = getattr(content, "parts", None)
                if parts:
                    texts = []
                    for part in parts:
                        if hasattr(part, "text") and part.text:
                            texts.append(part.text)
                    if texts:
                        return "\n".join(texts)

    raise ValueError("Gemini response did not return any text content.")


def _score_chunks(question: str, chunks_df: pd.DataFrame) -> list[int]:
    """
    Scores all chunks in ONE Gemini call.
    Returns list of int scores (0/1/2), same order as chunks_df rows.
    Falls back to all-zeros on any error so callers never crash.
    """
    chunks_text = ""
    for i, (_, row) in enumerate(chunks_df.iterrows()):
        chunks_text += f"\nChunk {i+1}:\n{str(row.get('text', ''))[:300]}\n"

    prompt = f"""
You are evaluating a retrieval system for CA (Chartered Accountancy) exams.

Question: {question}

Retrieved Chunks:
{chunks_text}

Score EACH chunk for relevance to the question.
Reply ONLY with a valid JSON array, one score per chunk. Example: [2, 1, 0, 1]

Scoring:
2 = directly relevant — chunk directly answers or is essential
1 = partially relevant — related but doesn't directly answer
0 = not relevant — unrelated

Rules:
- Return ONLY the JSON array
- Array length MUST match the number of chunks exactly
- No explanation, no markdown
"""

    response = None
    try:
        print("[GEMINI DEV CALL] metrics._score_chunks()")
        response = _get_model().generate_content(prompt)
        text     = _get_response_text(response).strip()
        text     = re.sub(r"```json|```", "", text).strip()

        scores = json.loads(text)

        if not isinstance(scores, list):
            raise ValueError("Response is not a JSON array")

        scores = [int(s) if s in (0, 1, 2) else 0 for s in scores]

        # Pad or trim to exactly match chunk count
        n      = len(chunks_df)
        scores = (scores + [0] * n)[:n]
        return scores

    except Exception as e:
        print(f"[metrics] _score_chunks failed: {e}")
        try:
            if response:
                raw = _get_response_text(response)
                print(f"[metrics] raw response: {raw[:200]}")
        except Exception:
            pass
        return [0] * len(chunks_df)


# ═══════════════════════════════════════════════════════════════════
# PUBLIC — evaluate_retrieval
# ═══════════════════════════════════════════════════════════════════

def evaluate_retrieval(
    question: str,
    chunks_df: pd.DataFrame,
    k: int = 4,
) -> Optional[dict]:
    """
    Returns full metrics dict when ENABLE_METRICS = True.
    Returns None when ENABLE_METRICS = False.
    api.py must check for None before accessing fields.
    """
    if not ENABLE_METRICS:
        return None

    scores     = _score_chunks(question, chunks_df)
    top_scores = scores[:k]

    high     = sum(1 for s in top_scores if s == 2)
    partial  = sum(1 for s in top_scores if s == 1)
    relevant = high + partial

    precision = relevant / k if k > 0 else 0.0

    if high >= 1:
        crag_status = "CORRECT"
        crag_action = "Good context found — answer should be reliable"
    elif partial >= 1:
        crag_status = "AMBIGUOUS"
        crag_action = "Partial context — answer may be incomplete"
    else:
        crag_status = "INCORRECT"
        crag_action = "No relevant chunks found — answer may be unreliable"

    crag_score = (high * 2 + partial) / (k * 2) if k > 0 else 0.0

    return {
        "precision_at_k": round(precision, 3),
        "crag_status":    crag_status,
        "crag_score":     round(crag_score, 3),
        "crag_action":    crag_action,
        "chunk_scores":   scores,
        "high":           high,
        "partial":        partial,
        "irrelevant":     sum(1 for s in top_scores if s == 0),
    }


# ═══════════════════════════════════════════════════════════════════
# PUBLIC — judge_answer
# ═══════════════════════════════════════════════════════════════════

def judge_answer(question: str, answer: str) -> Optional[dict]:
    """
    Returns judge dict when ENABLE_METRICS = True.
    Returns None when ENABLE_METRICS = False.
    """
    if not ENABLE_METRICS:
        return None

    prompt = f"""
You are a senior CA examiner evaluating a student's answer.

Question: {question}

Answer:
{answer[:1500]}

Rate this answer from 1 to 5.
Reply ONLY with this exact JSON format:
{{"score": 4, "reason": "brief one line reason"}}

Scoring guide:
5 = Complete, accurate, proper section references
4 = Mostly correct with minor gaps
3 = Partially correct, key points present
2 = Mostly incomplete or incorrect
1 = Wrong or irrelevant
"""

    response = None
    try:
        print("[GEMINI DEV CALL] metrics.judge_answer()")
        response = _get_model().generate_content(prompt)
        text     = _get_response_text(response).strip()
        text     = re.sub(r"```json|```", "", text).strip()

        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            text = match.group(0)

        parsed = json.loads(text)
        score  = max(1, min(5, int(parsed.get("score", 1))))
        reason = str(parsed.get("reason", ""))
        return {"score": score, "reason": reason}

    except Exception as e:
        print(f"[metrics] judge_answer failed: {e}")
        try:
            if response:
                raw = _get_response_text(response)
                for ch in raw:
                    if ch in ["1", "2", "3", "4", "5"]:
                        return {"score": int(ch), "reason": "score extracted from response"}
        except Exception:
            pass
        return {"score": 1, "reason": "Could not parse judge response"}
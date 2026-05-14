"""
search.py
─────────────────────────────────────────────────────────────────────
RAG retrieval with optional HyDE (toggled via dev_config.py).

ENABLE_HYDE = False  →  no Gemini; keyword classification only; raw query vector.
ENABLE_HYDE = True   →  one DEV Gemini call for HyDE text + classification;
                         blend hypothetical embedding with the question vector.
"""

import joblib
import numpy as np
import requests
import re
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

from dev_config import ENABLE_HYDE
from hyde import hypothetical_document       # always returns dict with classification

# ── Embedding corpora ─────────────────────────────────────────────
try:
    FOUNDATION_DF = joblib.load("embeddings_foundation.joblib")
    INTER_DF      = joblib.load("embeddings_Intermediate.joblib")
    FINAL_DF      = joblib.load("embeddings_Final.joblib")
except FileNotFoundError as e:
    raise RuntimeError(
        f"[search] Embedding file not found: {e}\n"
        "Make sure all three .joblib files are in the working directory."
    ) from e

ALL_DF  = pd.concat([FOUNDATION_DF, INTER_DF, FINAL_DF], ignore_index=True)
VECTORS = np.vstack(ALL_DF["embedding"].values)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def clean_text(text):
    """
    Sanitize text before sending to Ollama embedding.
    Removes NaN/None literals and collapses whitespace.
    Preserves Unicode (including ₹) — bge-m3 handles it fine.
    """
    if not text:
        return ""
    text = str(text)
    text = text.replace("NaN", "").replace("nan", "").replace("None", "").replace("null", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2000]


def embed(text):
    """Embed text using Ollama bge-m3. Always cleans input first."""
    text = clean_text(text)

    if not text:
        raise Exception("embed() received empty text after cleaning")

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": "bge-m3", "prompt": text},
            timeout=30,
        )
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise Exception(
            "Embedding service (Ollama) is not running. "
            "Start it with: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise Exception("Ollama embedding timed out. The model may still be loading.")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"Ollama returned an HTTP error: {e}")

    data = r.json()
    if "embedding" not in data:
        raise Exception(f"Embedding failed: {data}")

    return data["embedding"]


def rerank(query, chunks_df, top_n=4):
    pairs  = [[query, text] for text in chunks_df["text"].values]
    scores = reranker.predict(pairs)

    chunks_df = chunks_df.copy()
    chunks_df["rerank_score"] = scores

    return chunks_df.sort_values("rerank_score", ascending=False).head(top_n)


def is_numeric_query(q):
    """Used only when HyDE is on — determines blend weights."""
    return any(k in q.lower() for k in [
        "tax", "80c", "80d", "income", "salary",
        "gst", "compute", "calculate",
    ])


def search_rag(question, top_k=8, top_n=4, doc_retrieval_hint: str | None = None):
    """
    question — primary search text (the clean user question).
               Never pass the full combined query with memory/OCR here;
               that string is for process_query() only.

    doc_retrieval_hint — optional short factual line (e.g. from uploaded bank
    / tax JSON). When set, it is appended for the primary embedding and passed
    into HyDE (when enabled) so the hypothetical paragraph stays consistent
    with upload-derived numbers.

    HyDE / classification:
      hypothetical_document() always returns a dict with `classification`.
      Blend weights use classification.needs_calculation when present.

    Returns:
        (reranked_chunks_df, classification_dict)
    """
    print("\n[RAG] Starting retrieval...")

    embed_question = question.strip()
    if doc_retrieval_hint and str(doc_retrieval_hint).strip():
        hint = clean_text(str(doc_retrieval_hint).strip())
        if hint:
            embed_question = clean_text(f"{embed_question}\n{hint}")[:2000]

    # ── Step 1: Embed question (+ optional upload hint) ───────────
    raw_vec = np.array(embed(embed_question))

    # ── Step 2: HyDE blend (hypothetical text only when HyDE on and Gemini succeeds) ─
    hyde_response = hypothetical_document(question.strip(), context_hint=doc_retrieval_hint)
    hyde_text = hyde_response.get("hypothetical_document")
    classification = hyde_response.get("classification") or {}

    is_numeric = (
        classification.get("needs_calculation", False)
        if classification
        else is_numeric_query(question.strip())
    )

    if hyde_text:
        hyde_vec = np.array(embed(hyde_text))

        if is_numeric:
            print("[HyDE] Numeric / calc intent → RAW priority (0.8 / 0.2)")
            q_vec = (0.8 * raw_vec) + (0.2 * hyde_vec)
        else:
            print("[HyDE] Theory intent → HyDE priority (0.4 / 0.6)")
            q_vec = (0.4 * raw_vec) + (0.6 * hyde_vec)
    else:
        if ENABLE_HYDE:
            print("[HyDE] No hypothetical text — raw vector only")
        q_vec = raw_vec

    # ── Step 3: Cosine similarity over full corpus ────────────────
    sims    = cosine_similarity(VECTORS, [q_vec]).flatten()
    top_idx = sims.argsort()[::-1][:top_k]

    candidates               = ALL_DF.iloc[top_idx].copy()
    candidates["similarity"] = sims[top_idx]

    print(f"\n[Retrieval] Top {top_k}:")
    print(candidates[["level", "book", "chunk_id", "similarity"]])

    # ── Step 4: Rerank ────────────────────────────────────────────
    reranked = rerank(question.strip(), candidates, top_n=top_n)

    print(f"\n[Reranker] Top {top_n}:")
    print(reranked[["level", "book", "chunk_id", "rerank_score"]])

    return reranked, classification
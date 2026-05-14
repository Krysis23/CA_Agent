import json
import re
import time
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


def generate_with_backoff(prompt_text, retries=3):
    """Call Gemini with automatic exponential-backoff retry on 429 rate-limit errors."""
    for attempt in range(retries):
        try:
            print(f"[GEMINI CALL] ca_agent.generate_with_backoff() (attempt {attempt+1})")
            return model.generate_content(prompt_text).text
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"[ca_agent] 429 rate limit hit, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise




def detect_query_type(query):
    q = query.lower()
    if any(k in q for k in ["gst", "input tax credit", "output tax"]):
        return "gst"
    elif any(k in q for k in ["tax", "80c", "80d", "income", "salary"]):
        return "income_tax"
    else:
        return "theory"


def has_calculation_intent(query):
    q = query.lower()
    calc_keywords = [
        "calculate", "calculation", "compute", "work out",
        "tax payable", "how much tax", "net gst", "liability",
        "show working", "step by step", "numerical", "solve",
    ]
    return any(k in q for k in calc_keywords)


def detect_tax_regime(query):
    q = query.lower()
    if "new regime" in q:
        return "new"
    return "old"




def safe(x):
    return x if isinstance(x, (int, float)) else 0




def calculate_tax(data, regime="old"):
    income = (
        safe(data.get("salary"))
        + safe(data.get("house_property"))
        + safe(data.get("business_income"))
        + safe(data.get("unexplained_income"))
        + safe(data.get("other_income", 0))
    )

    if regime == "old":
        d80c               = min(safe(data.get("80c")), 150000)
        d80d_self          = min(safe(data.get("80d")), 25000)
        d80d_parents       = min(safe(data.get("80d_parents", 0)), 50000)  # senior citizen limit
        standard_deduction = 50000

        taxable_income = max(
            0,
            income
            - d80c
            - d80d_self
            - d80d_parents
            - standard_deduction
            - safe(data.get("house_property_loss", 0))   # subtract the loss amount
        )

        if taxable_income <= 250000:
            tax = 0
        elif taxable_income <= 500000:
            tax = (taxable_income - 250000) * 0.05
        elif taxable_income <= 1000000:
            tax = (250000 * 0.05) + (taxable_income - 500000) * 0.20
        else:
            tax = (250000 * 0.05) + (500000 * 0.20) + (taxable_income - 1000000) * 0.30

        
        if taxable_income <= 500000:
            tax = 0

        
        surcharge = 0
        if taxable_income > 5000000:
            surcharge = tax * 0.10
        if taxable_income > 10000000:
            surcharge = tax * 0.15
        if taxable_income > 20000000:
            surcharge = tax * 0.25
        if taxable_income > 50000000:
            surcharge = tax * 0.37

        tax += surcharge

        cess = tax * 0.04

    else:  # new regime — FY 2025-26 (Section 87A: full rebate up to ₹12L; marginal relief ₹12L–₹12.75L)
        std_ded = 75000 if safe(data.get("salary", 0)) > 0 else 0
        taxable_income = max(0, income - std_ded)

        if taxable_income <= 400000:
            slab_tax = 0.0
        elif taxable_income <= 800000:
            slab_tax = (taxable_income - 400000) * 0.05
        elif taxable_income <= 1200000:
            slab_tax = 20000 + (taxable_income - 800000) * 0.10
        elif taxable_income <= 1600000:
            slab_tax = 60000 + (taxable_income - 1200000) * 0.15
        elif taxable_income <= 2000000:
            slab_tax = 120000 + (taxable_income - 1600000) * 0.20
        elif taxable_income <= 2400000:
            slab_tax = 200000 + (taxable_income - 2000000) * 0.25
        else:
            slab_tax = 300000 + (taxable_income - 2400000) * 0.30

        # Section 87A (FY 2025-26 new regime): full rebate of slab tax if TI ≤ ₹12L → net tax before cess = 0
        if taxable_income <= 1200000:
            rebate_87a = slab_tax
            base_tax = max(0.0, slab_tax - rebate_87a)
        elif taxable_income <= 1275000:
            base_tax = min(slab_tax, taxable_income - 1200000)
            rebate_87a = max(0.0, slab_tax - base_tax)
        else:
            rebate_87a = 0.0
            base_tax = slab_tax

        surcharge = 0.0
        if taxable_income > 20000000:
            surcharge = base_tax * 0.25
        elif taxable_income > 10000000:
            surcharge = base_tax * 0.15
        elif taxable_income > 5000000:
            surcharge = base_tax * 0.10

        # Cess 4% on post–Section 87A base tax only (FY 2025-26); surcharge sits outside cess base per stated rules
        tax = base_tax + surcharge
        cess = base_tax * 0.04

    return {
        "income":         income,
        "taxable_income": taxable_income,
        "tax":            round(tax, 2),
        "cess":           round(cess, 2),
        "total_tax":      round(tax + cess, 2),
    }


def calculate_gst(data):
    output_tax = safe(data.get("output_tax"))
    input_tax  = safe(data.get("input_tax"))

    return {
        "output_tax":      output_tax,
        "input_tax":       input_tax,
        "net_gst_payable": max(0, output_tax - input_tax),
    }




def clean_data(data):
    for key in data:
        if data[key] is None:
            data[key] = 0
    return data


def extract_data(query):
    query = query[:2000]

    prompt = f"""
Extract financial data from the query.
Return ONLY valid JSON, no markdown, no explanation.

Extract ALL relevant financial signals visible in the query.
Do not restrict to only common fields.

Return keys (use 0 if missing):
- salary
- business_income
- house_property
- house_property_loss
- unexplained_income
- other_income
- total_credits
- total_debits
- additional_info_amount
- 80c
- 80d
- 80d_parents
- output_tax
- input_tax

Query:
{query}
"""
    print("[GEMINI CALL] ca_agent.extract_data()")
    response = model.generate_content(prompt)
    text     = response.text.strip()
    text     = re.sub(r"```json|```", "", text).strip()

    try:
        data = json.loads(text)
        return clean_data(data)
    except Exception:
        print("Extraction Error:", text)
        return {}


def extract_data_from_document(doc):
    if not doc:
        return {}

    is_new = "person_entity" in doc
    is_api_bank = doc.get("document_type") == "bank_statement" and "person_name" in doc

    if is_new or is_api_bank:
        chat = doc.get("chat_context") or {}
        inc = chat.get("income") or {}
        bank_chat = chat.get("bank_summary") or {}
        bank_root = doc.get("bank") or {}
        tb = doc.get("tax_breakdown") or {}

        business_income = (
            safe(inc.get("business_income", 0))
            or safe(doc.get("estimated_annual_income", 0))
        )

        tc = safe(doc.get("total_credits", 0)) or safe(bank_root.get("total_credits", 0)) or safe(bank_chat.get("credits", 0))
        td = safe(doc.get("total_debits", 0)) or safe(bank_root.get("total_debits", 0)) or safe(bank_chat.get("debits", 0))
        ob = safe(doc.get("opening_balance", 0)) or safe(bank_root.get("opening_balance", 0)) or safe(bank_chat.get("opening_balance", 0))
        cl = safe(doc.get("closing_balance", 0)) or safe(bank_root.get("closing_balance", 0)) or safe(bank_chat.get("closing_balance", 0))

        person = str(doc.get("person_entity", "") or "") if is_new else str(doc.get("person_name", "") or "")
        period = str(doc.get("statement_period", "") or "") if is_new else str(doc.get("period", "") or "")
        summary = str(doc.get("summary_text", "") or "") if is_new else str(doc.get("raw_text_summary", "") or "")

        return {
            "salary":              safe(inc.get("salary", 0)),
            "business_income":     business_income,
            "house_property":      0,
            "house_property_loss": 0,
            "unexplained_income":  0,
            "other_income":        safe(inc.get("other_income", 0)),
            "80c":                 0,
            "80d":                 0,
            "80d_parents":         0,
            "output_tax":          0,
            "input_tax":           0,
            "total_credits":       tc,
            "total_debits":        td,
            "opening_balance":     ob,
            "closing_balance":     cl,
            "document_type":       "bank_statement",
            "person_name":         person,
            "period":              period,
            "raw_text_summary":    summary,
            "slab_breakdown":      tb.get("slab_breakdown", []),
            "gemini_total_tax":    safe(doc.get("estimated_tax_new_regime_fy_2025_26", 0)),
            "gemini_taxable":      safe(doc.get("taxable_income_estimate", 0)),
        }

    income = doc.get("income") or {}
    bank = doc.get("bank") or {}
    gst = doc.get("gst") or {}
    deductions = doc.get("deductions") or {}
    est = safe(doc.get("estimated_annual_income", 0))
    business = safe(income.get("business_income", 0)) or est

    return {
        "salary":              safe(income.get("salary", 0)),
        "business_income":     business,
        "house_property":      safe(income.get("rental_income", 0)),
        "house_property_loss": safe(income.get("house_property", 0)),
        "unexplained_income":  0,
        "other_income":        safe(income.get("other_income", 0)),
        "80c":                 safe(deductions.get("80c", 0)),
        "80d":                 safe(deductions.get("80d", 0)),
        "80d_parents":         0,
        "output_tax":          safe(gst.get("output_tax", 0)),
        "input_tax":           safe(gst.get("input_tax", 0)),
        "total_credits":       safe(bank.get("total_credits", 0)),
        "total_debits":        safe(bank.get("total_debits", 0)),
        "opening_balance":     safe(bank.get("opening_balance", 0)),
        "closing_balance":     safe(bank.get("closing_balance", 0)),
        "document_type":       doc.get("document_type", "other"),
        "person_name":         str(doc.get("person_name", "") or ""),
        "period":              str(doc.get("period", "") or ""),
        "raw_text_summary":    str(doc.get("raw_text_summary", "") or ""),
        "slab_breakdown":      [],
        "gemini_total_tax":    0,
        "gemini_taxable":      0,
    }


def tax_appendix_from_doc_data(doc_data: dict) -> str:
    """Readable slab / tax lines from persisted upload JSON (for LLM doc_context)."""
    if not doc_data or not isinstance(doc_data, dict):
        return ""
    lines: list[str] = []
    tb = doc_data.get("tax_breakdown")
    if isinstance(tb, dict) and tb:
        lines.append("— Upload tax model (statement extraction):")
        if safe(tb.get("gross_income")) > 0:
            lines.append(f"  Gross income: ₹{safe(tb.get('gross_income')):,.2f}")
        if safe(tb.get("standard_deduction")) > 0:
            lines.append(f"  Standard deduction: ₹{safe(tb.get('standard_deduction')):,.2f}")
        if safe(tb.get("taxable_income")) > 0:
            lines.append(f"  Taxable income: ₹{safe(tb.get('taxable_income')):,.2f}")
        if safe(tb.get("rebate_87a")) > 0:
            lines.append(f"  Rebate 87A: ₹{safe(tb.get('rebate_87a')):,.2f}")
        if safe(tb.get("base_tax")) > 0 or safe(tb.get("cess_4_percent")) > 0:
            lines.append(
                f"  Base tax: ₹{safe(tb.get('base_tax')):,.2f}; "
                f"Cess 4%: ₹{safe(tb.get('cess_4_percent')):,.2f}"
            )
        if safe(tb.get("final_tax")) > 0:
            lines.append(f"  Final tax: ₹{safe(tb.get('final_tax')):,.2f}")
        slabs = tb.get("slab_breakdown")
        if isinstance(slabs, list) and slabs:
            lines.append("  Slab-wise (model):")
            for s in slabs:
                if not isinstance(s, dict):
                    continue
                lines.append(
                    f"    {s.get('range', '?')} @ {s.get('rate', '?')}: ₹{safe(s.get('tax')):,.2f}"
                )
    est = safe(doc_data.get("estimated_annual_income", 0))
    if est > 0:
        lines.append(f"— Estimated annual income (model): ₹{est:,.2f}")
    etop = safe(doc_data.get("estimated_tax_new_regime_fy_2025_26", 0))
    if etop > 0 and not (isinstance(tb, dict) and safe(tb.get("final_tax")) > 0):
        lines.append(f"— Estimated tax (top-level field): ₹{etop:,.2f}")
    if not lines:
        return ""
    return "\n" + "\n".join(lines) + "\n"


def _is_bank_statement_doc(doc_data: dict | None) -> bool:
    """Uploaded bank statement from vision pipeline — Gemini figures are authoritative (no Python tax engine)."""
    if not doc_data or not isinstance(doc_data, dict):
        return False
    return str(doc_data.get("document_type", "") or "").strip().lower() == "bank_statement"


def process_query(query, retrieved_context, doc_data=None, classification: dict | None = None):
    """
    query               — user's question (may include document blob)
    retrieved_context   — ICAI chunks from search_rag()
    doc_data            — optional structured upload from Gemini Vision
    classification   — optional HyDE / keyword routing from search_rag()
    """

    user_query = query.split("\n")[0]

    if classification and isinstance(classification, dict):
        tt = classification.get("tax_type")
        if tt in ("income_tax", "gst"):
            query_type = tt
        else:
            query_type = detect_query_type(user_query)
        calculation_intent = bool(classification.get("needs_calculation"))
        r = classification.get("regime")
        regime = r if r in ("old", "new") else detect_tax_regime(user_query)
    else:
        query_type = detect_query_type(user_query)
        calculation_intent = has_calculation_intent(user_query)
        regime = detect_tax_regime(user_query)

    query             = query[:3000]
    retrieved_context = retrieved_context[:12000]

    computed_result = None

    want_extract = calculation_intent or has_calculation_intent(user_query)

    if doc_data:
        data = extract_data_from_document(doc_data)
        print("[ca_agent] Using Gemini Vision extracted data:", data)
    elif query_type != "theory" and want_extract:
        data = extract_data(user_query)
        print("[ca_agent] Extracted from query:", data)
    else:
        data = {}

    if not doc_data:
        if classification and isinstance(classification, dict):
            early_icai_only = (
                classification.get("query_type") == "theory"
                and not classification.get("needs_calculation", False)
                and classification.get("tax_type") == "none"
            )
        else:
            early_icai_only = query_type == "theory"
        if early_icai_only:
            prompt = f"""
You are a Chartered Accountant examiner.
Use ONLY ICAI content below to answer.

ICAI Content:
{retrieved_context}

Question:
{query}

Answer in structured bullet format with section references where possible.
"""
            return generate_with_backoff(prompt)

    # Broaden calc intent for non–bank-statement docs only (bank uploads never use the Python tax engine).
    if doc_data and not calculation_intent and not _is_bank_statement_doc(doc_data):
        calculation_intent = any(
            k in user_query.lower()
            for k in [
                "breakdown", "break down", "give me", "show me", "calculate",
                "computation", "compute", "how much", "tax", "slab", "detail",
                "working", "liability", "estimate", "calculation",
            ]
        )
        if calculation_intent and query_type == "theory":
            query_type = "income_tax"

    bank_stmt = _is_bank_statement_doc(doc_data)

    if not bank_stmt:
        if calculation_intent and query_type in ("income_tax", "theory") and doc_data:
            computed_result = calculate_tax(data, regime)
            print("[ca_agent] Tax computed:", computed_result)

        elif calculation_intent and query_type == "income_tax":
            computed_result = calculate_tax(data, regime)
            print("[ca_agent] Tax computed:", computed_result)

        elif calculation_intent and query_type == "gst":
            computed_result = calculate_gst(data)
            print("[ca_agent] GST computed:", computed_result)

    
    doc_context = ""
    if doc_data:
        tax_x = tax_appendix_from_doc_data(doc_data)
        doc_context = f"""
Document Type   : {data.get('document_type', 'N/A')}
Person / Entity : {data.get('person_name', 'N/A')}
Period          : {data.get('period', 'N/A')}
Summary         : {data.get('raw_text_summary', 'N/A')}
Opening balance : {data.get('opening_balance', 0)}
Total Credits   : {data.get('total_credits', 0)}
Total Debits    : {data.get('total_debits', 0)}
Closing balance : {data.get('closing_balance', 0)}
{tax_x}"""

    bank_authority_rules = ""
    if bank_stmt and doc_context.strip():
        bank_authority_rules = """
- UPLOADED BANK STATEMENT: Every balance, credit, debit, estimated income, taxable income, slab-wise tax, and total tax figure in "Document Context" was produced by the document extraction model (Gemini). These numbers are FINAL and AUTHORITATIVE for this statement.
- Do NOT recompute, replace, or "correct" them using the in-app tax engine, generic slab tables, or your own arithmetic. Do not present a parallel Python/calculator result.
- You may explain, interpret, or relate those figures to ICAI material, but any numeric answer about this statement must match Document Context exactly.
"""

    if computed_result is not None:
        prompt = f"""
You are a Chartered Accountant examiner and tutor.

ICAI Content:
{retrieved_context}

{f"Document Context:{doc_context}" if doc_context else ""}

Student Question:
{query}

Computed Result (FINAL — DO NOT recompute or change any numbers):
{computed_result}

STRICT RULES:
- Computed result is FINAL and AUTHORITATIVE
- DO NOT recompute, DO NOT change any number
- Use ICAI content for explanation wording and section references WHEN relevant.
- If retrieved ICAI context is unrelated/insufficient, still answer using the computed result and accepted tax principles.
-DO NOT invent subsection numbers unless explicitly present in ICAI content.
- Always show:
  1. Income computation (with sources)
  2. Deductions applied
  3. Tax calculation step by step
  4. Final tax payable
- If you use tables, output VALID markdown tables only:
  - Keep header, separator, and each row on a NEW line
  - Do NOT merge multiple rows into one paragraph

Answer clearly with steps.
"""
    else:
        prompt = f"""
You are a Chartered Accountant examiner and tutor.

ICAI Content:
{retrieved_context}

{f"Document Context:{doc_context}" if doc_context else ""}

Student Question:
{query}

STRICT RULES:
- This is a conceptual/theory response unless user explicitly asks to calculate.
- Do NOT show tax/GST computation blocks, formulas, or zero-valued calculation tables unless explicitly asked.
- Use ICAI content ONLY for explanation wording and section references.
- DO NOT invent subsection numbers unless explicitly present in ICAI content.
- Keep answer focused, structured, and concise.
- If you use tables, output VALID markdown tables only:
  - Keep header, separator, and each row on a NEW line
  - Do NOT merge multiple rows into one paragraph
{bank_authority_rules}
Answer clearly in bullet points.
"""

    return generate_with_backoff(prompt)
import os
import uuid
import json
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory_store = {
    "batches": {},
    "confirmed": []
}


# ── helpers ──────────────────────────────────────────────────────────────────

def get_client(api_key: str | None) -> genai.Client:
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing Gemini API key. Please reload and enter your key.")
    return genai.Client(api_key=api_key)

def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def signal_to_text(signal: dict) -> str:
    prefix = f"[{signal['source_type']}] {signal['source_name']}: "
    c = signal["content"]
    st = signal["signal_type"]
    if st == "decision":
        return prefix + f"{c.get('text', '')}. Rationale: {c.get('rationale', '')}"
    elif st == "component":
        constraints = ", ".join(c.get("constraints", []))
        return prefix + f"{c.get('name', '')}: {c.get('description', '')}. Constraints: {constraints}"
    elif st == "pattern":
        return prefix + f"{c.get('name', '')}: {c.get('description', '')}"
    elif st == "constraint":
        return prefix + f"{c.get('text', '')} (type: {c.get('type', '')}, severity: {c.get('severity', '')})"
    elif st == "open_question":
        return prefix + f"{c.get('question', '')}. Context: {c.get('context', '')}"
    elif st == "persona":
        needs = ", ".join(c.get("needs", []))
        pain_points = ", ".join(c.get("pain_points", []))
        return prefix + f"{c.get('name', '')}. Needs: {needs}. Pain points: {pain_points}"
    elif st == "key_fact":
        return prefix + f"{c.get('text', '')} [category: {c.get('category', '')}]"
    return prefix + json.dumps(c)


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve_relevant(question: str, client: genai.Client, top_k: int = 15):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    q_vec = result.embeddings[0].values
    scored = []
    for signal in memory_store["confirmed"]:
        score = cosine_similarity(q_vec, signal["embedding"])
        scored.append((score, signal))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, signal in scored[:top_k]:
        s = {k: v for k, v in signal.items() if k != "embedding"}
        s["_score"] = round(score, 4)
        out.append(s)
    return out


def safe_parse_extraction(raw: str, source_name: str, source_type: str, batch_id: str):
    try:
        data = json.loads(strip_fences(raw))
    except Exception:
        data = {
            "signals": [{"temp_id": "t1", "signal_type": "key_fact",
                         "content": {"text": raw[:2000], "category": "parse_error"}}],
            "relationships": [],
            "extraction_notes": "Parse failed — stored raw response as key_fact"
        }

    now = datetime.utcnow().isoformat() + "Z"
    temp_to_real = {}
    signals = []
    for s in data.get("signals", []):
        sid = str(uuid.uuid4())
        temp_id = s.get("temp_id", "")
        temp_to_real[temp_id] = sid
        signal = {
            "id": sid,
            "batch_id": batch_id,
            "signal_type": s.get("signal_type", "key_fact"),
            "content": s.get("content", {}),
            "source_name": source_name,
            "source_type": source_type,
            "status": "pending",
            "ingested_at": now,
            "embedding": [],
            "relationships": []
        }
        signals.append(signal)

    relationships = []
    for r in data.get("relationships", []):
        from_id = temp_to_real.get(r.get("from_temp_id", ""))
        to_id = temp_to_real.get(r.get("to_temp_id", ""))
        if from_id and to_id:
            rel = {"target_id": to_id, "relationship_type": r.get("relationship_type", "")}
            for s in signals:
                if s["id"] == from_id:
                    s["relationships"].append(rel)
            relationships.append({
                "from_id": from_id,
                "to_id": to_id,
                "relationship_type": r.get("relationship_type", "")
            })

    return signals, relationships, data.get("extraction_notes", "")


def summarise(signals):
    by_type = {}
    for s in signals:
        by_type[s["signal_type"]] = by_type.get(s["signal_type"], 0) + 1
    return {"total": len(signals), "by_type": by_type}


def strip_embeddings(signals):
    return [{k: v for k, v in s.items() if k != "embedding"} for s in signals]


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r") as f:
        return f.read()


@app.post("/validate-key")
@limiter.limit("5/minute")
async def validate_key(request: Request, x_gemini_api_key: str | None = Header(default=None)):
    client = get_client(x_gemini_api_key)
    try:
        client.models.embed_content(
            model="gemini-embedding-001",
            contents="test",
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
    except genai_errors.ClientError as e:
        raise HTTPException(status_code=401, detail=f"Invalid API key: {e}")
    return {"valid": True}


@app.post("/extract")
@limiter.limit("10/minute")
async def extract(
    request: Request,
    source_name: str = Form(...),
    source_type: str = Form(...),
    text: Optional[str] = Form(None),
    files: list[UploadFile] = File(default=[]),
    x_gemini_api_key: str | None = Header(default=None)
):
    client = get_client(x_gemini_api_key)
    parts = []

    extraction_prompt = f"""
You are extracting structured product and design context signals from the
following {source_type} source titled '{source_name}'.

Extract every signal you can find. Be thorough. Capture not just what was
decided but WHY — rationale is as important as the decision itself.

For any images provided: describe UI components visible, layout patterns,
states shown, any text or labels readable. Treat visual evidence with the
same weight as written decisions.

Identify relationships between signals where they exist
(e.g. a decision constrains a component, a persona drives an open_question).

Use these signal types only:
  decision, component, pattern, constraint, open_question, persona, key_fact

Content shape per type:
  decision:      {{ "text": "...", "rationale": "...", "confidence": "high|medium|low" }}
  component:     {{ "name": "...", "description": "...", "constraints": ["..."] }}
  pattern:       {{ "name": "...", "description": "...", "examples": ["..."] }}
  constraint:    {{ "text": "...", "type": "technical|business|design", "severity": "hard|soft" }}
  open_question: {{ "question": "...", "context": "...", "impact": "high|medium|low" }}
  persona:       {{ "name": "...", "needs": ["..."], "pain_points": ["..."] }}
  key_fact:      {{ "text": "...", "category": "..." }}

Return ONLY a raw JSON object. No markdown fences. No preamble. No explanation.
Schema:
{{
  "signals": [
    {{ "temp_id": "t1", "signal_type": "...", "content": {{...}} }},
    ...
  ],
  "relationships": [
    {{ "from_temp_id": "t1", "to_temp_id": "t2", "relationship_type": "..." }}
  ],
  "extraction_notes": "brief summary of what you found and what was ambiguous"
}}

Content to extract from:
"""

    if text:
        parts.append(extraction_prompt + "\n\n" + text)
    else:
        parts.append(extraction_prompt)

    for upload in files:
        raw = await upload.read()
        fname = upload.filename or ""
        if fname.lower().endswith(".pdf"):
            try:
                import fitz
                doc = fitz.open(stream=raw, filetype="pdf")
                pdf_text = "\n".join(page.get_text() for page in doc)
                parts.append(f"\n[PDF: {fname}]\n{pdf_text}")
            except Exception as e:
                parts.append(f"\n[PDF: {fname} — extraction error: {e}]")
        elif fname.lower().endswith((".png", ".jpg", ".jpeg")):
            mime = "image/png" if fname.lower().endswith(".png") else "image/jpeg"
            parts.append(types.Part.from_bytes(data=raw, mime_type=mime))

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=parts)
    except genai_errors.ClientError as e:
        if e.code == 429:
            raise HTTPException(
                status_code=429,
                detail="Gemini API quota exhausted. Please wait a minute and try again, or upgrade your Google AI plan."
            )
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")
    raw_text = response.text

    batch_id = str(uuid.uuid4())
    signals, relationships, notes = safe_parse_extraction(raw_text, source_name, source_type, batch_id)
    memory_store["batches"][batch_id] = signals

    return {
        "batch_id": batch_id,
        "signals": strip_embeddings(signals),
        "relationships": relationships,
        "extraction_notes": notes,
        "summary": summarise(signals)
    }


class ReviseRequest(BaseModel):
    batch_id: str
    feedback: str


@app.post("/revise")
@limiter.limit("10/minute")
async def revise(request: Request, body: ReviseRequest, x_gemini_api_key: str | None = Header(default=None)):
    client = get_client(x_gemini_api_key)
    current_signals = memory_store["batches"].get(body.batch_id)
    if current_signals is None:
        raise HTTPException(status_code=404, detail="Batch not found")

    source_name = current_signals[0]["source_name"] if current_signals else "unknown"
    source_type = current_signals[0]["source_type"] if current_signals else "other"

    revision_prompt = f"""
You previously extracted signals from a document. The user has reviewed them
and provided feedback. Revise the signals accordingly.

Original signals:
{json.dumps(strip_embeddings(current_signals), indent=2)}

User feedback:
{body.feedback}

Apply the feedback precisely. You may add new signals, remove signals,
modify existing ones, or update relationships.

Return ONLY a raw JSON object in the exact same schema as the original extraction.
No markdown fences. No preamble.
{{
  "signals": [...],
  "relationships": [...],
  "extraction_notes": "what you changed and why"
}}
"""

    response = client.models.generate_content(model="gemini-2.5-flash", contents=revision_prompt)
    raw_text = response.text

    signals, relationships, notes = safe_parse_extraction(raw_text, source_name, source_type, body.batch_id)
    memory_store["batches"][body.batch_id] = signals

    return {
        "batch_id": body.batch_id,
        "signals": strip_embeddings(signals),
        "relationships": relationships,
        "extraction_notes": notes,
        "summary": summarise(signals)
    }


class ConfirmRequest(BaseModel):
    batch_id: str


@app.post("/confirm")
@limiter.limit("20/minute")
async def confirm(request: Request, body: ConfirmRequest, x_gemini_api_key: str | None = Header(default=None)):
    client = get_client(x_gemini_api_key)
    signals = memory_store["batches"].get(body.batch_id)
    if signals is None:
        raise HTTPException(status_code=404, detail="Batch not found")

    for signal in signals:
        text = signal_to_text(signal)
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        signal["embedding"] = result.embeddings[0].values
        signal["status"] = "confirmed"

    memory_store["confirmed"].extend(signals)
    del memory_store["batches"][body.batch_id]

    sources = list({s["source_name"] for s in memory_store["confirmed"]})
    return {
        "confirmed_count": len(signals),
        "total_memory_size": len(memory_store["confirmed"]),
        "sources_in_memory": sources
    }


@app.get("/memory")
async def get_memory():
    confirmed = memory_store["confirmed"]
    by_type = {}
    for s in confirmed:
        by_type[s["signal_type"]] = by_type.get(s["signal_type"], 0) + 1

    signals_by_type = {}
    for s in confirmed:
        st = s["signal_type"]
        if st not in signals_by_type:
            signals_by_type[st] = []
        signals_by_type[st].append({k: v for k, v in s.items() if k != "embedding"})

    relationship_graph = []
    for s in confirmed:
        for rel in s.get("relationships", []):
            relationship_graph.append({
                "from_id": s["id"],
                "to_id": rel["target_id"],
                "relationship_type": rel["relationship_type"]
            })

    sources = list({s["source_name"] for s in confirmed})

    return {
        "total": len(confirmed),
        "by_type": by_type,
        "sources": sources,
        "signals_by_type": signals_by_type,
        "relationship_graph": relationship_graph
    }


class QueryRequest(BaseModel):
    question: str
    mode: str  # "answer" | "generate_artifact"


@app.post("/query")
@limiter.limit("20/minute")
async def query(request: Request, body: QueryRequest, x_gemini_api_key: str | None = Header(default=None)):
    client = get_client(x_gemini_api_key)
    if not memory_store["confirmed"]:
        raise HTTPException(status_code=400, detail="No confirmed memory yet")

    retrieved = retrieve_relevant(body.question, client, top_k=15)
    retrieved_signals = [{k: v for k, v in s.items() if k not in ("embedding",)} for s in retrieved]

    if body.mode == "answer":
        prompt = f"""
You are a product intelligence engine. Answer the user's question using ONLY
the confirmed memory signals below as your source of truth.
These signals were retrieved semantically — they are the most relevant signals
to this question from the full memory.

Retrieved memory signals:
{json.dumps(retrieved_signals, indent=2)}

Question: {body.question}

Rules:
- Cite which signals informed your answer (by signal_type and source_name)
- Flag any conflicts or contradictions you see across signals
- If the retrieved signals don't contain enough information, say so explicitly
- Do not invent information not present in the signals

Return ONLY a raw JSON object. No markdown fences. No preamble.
{{
  "answer": "...",
  "signals_used": [
    {{ "signal_type": "...", "source_name": "...", "content_summary": "..." }}
  ],
  "confidence": "high|medium|low",
  "conflicts": ["..."]
}}
"""
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw_text = response.text
        try:
            data = json.loads(strip_fences(raw_text))
        except Exception:
            data = {"answer": raw_text, "signals_used": [], "confidence": "low", "conflicts": []}

        retrieval_scores = [{"signal_id": s["id"], "score": s["_score"]} for s in retrieved]
        return {
            "answer": data.get("answer", ""),
            "signals_used": data.get("signals_used", []),
            "confidence": data.get("confidence", "low"),
            "conflicts": data.get("conflicts", []),
            "retrieval_scores": retrieval_scores,
            "retrieved_signals": retrieved_signals
        }

    else:  # generate_artifact
        prompt = f"""
You are a React component generator with deep product context.
Using the retrieved memory signals below as your sole source of truth,
generate a complete working React component for the following request.
These signals were retrieved semantically as the most relevant to this request.

Request: {body.question}

Retrieved memory signals:
{json.dumps(retrieved_signals, indent=2)}

Requirements for the generated component:
- Use ONLY patterns, components, and constraints found in the signals
- Respect ALL hard constraints found in the signals
- Use Tailwind CSS classes for styling
- Include useState/useEffect as needed
- Export default the component
- At the very top of the file include this comment block exactly:

  /*
   * MEMORY SIGNALS APPLIED
   * ----------------------
   * [signal_type] source_name: what this signal drove in the design
   * [signal_type] source_name: what this signal drove in the design
   * ...
   */

Return ONLY the raw React component code. No markdown fences. No explanation.
"""
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        code = response.text
        if code.startswith("```"):
            code = strip_fences(code)

        retrieval_scores = [{"signal_id": s["id"], "score": s["_score"]} for s in retrieved]

        decisions_applied = []
        for s in retrieved_signals:
            if s["signal_type"] in ("decision", "constraint"):
                c = s["content"]
                decisions_applied.append(c.get("text", c.get("name", "")))

        return {
            "code": code,
            "signals_used": retrieved_signals,
            "decisions_applied": decisions_applied,
            "retrieval_scores": retrieval_scores
        }


@app.delete("/memory")
@limiter.limit("5/minute")
async def clear_memory(request: Request):
    memory_store["confirmed"].clear()
    memory_store["batches"].clear()
    return {"cleared": True}
  

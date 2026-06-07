# Context Memory Prototype

## What it does
Ingests product context (PRDs, Slack threads, design notes, screenshots),
extracts structured signals via Gemini, lets you review and revise before
confirming, then uses semantic similarity to retrieve relevant signals when
you ask questions or request React component generation.

## Setup
1. Copy files into a folder
2. Create .env file: `GEMINI_API_KEY=your_key_here`
   Get a free key: https://aistudio.google.com/app/apikey
3. `pip install -r requirements.txt`
4. `uvicorn main:app --reload`
5. Open http://localhost:8000

## Test flow
1. Paste a PRD or upload a PDF + screenshots → Extract signals
2. Review the signal map — click nodes to inspect
3. Give feedback if signals are wrong → Gemini revises → review again
4. Confirm → signals embedded and stored in memory
5. Ingest another source (Slack thread, design notes) → memory accumulates
6. Switch to Query panel → Ask a question or generate a React component
7. See which signals scored highest in semantic retrieval

## Architecture decisions
**Memory:** In-process Python dict. Wipes on restart. Intentional for prototype.

**Retrieval:** Gemini text-embedding-004 → cosine similarity → top 15 signals.
Each signal embedded at confirm time. Query embedded at retrieval time.
No vector DB — numpy cosine similarity over ~50–200 signals is instant.

**Generation:** Top 15 retrieved signals stuffed into Gemini 2.0 Flash prompt.
Full context stuffing works here because retrieval already filtered.

## Stack
FastAPI + uvicorn | google-generativeai | pymupdf (PDF extraction) | numpy

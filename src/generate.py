"""
generate.py
-----------
Step 4 of the RAG pipeline: construct a grounded prompt from the
retrieved chunks and call an LLM to produce a short, source-grounded
answer.

LLM: Google Gemini API (gemini-2.5-flash)
Why Gemini: Highly capable, extremely fast, and has a generous free tier.
Requires a GEMINI_API_KEY environment variable. You can get a free key from:
https://aistudio.google.com/app/apikey
"""
import os
from google import genai
from google.genai import types

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are a Q&A assistant that answers questions strictly using the "
    "provided context, which is extracted from a Press Information Bureau "
    "(PIB) backgrounder on India's health transformation. "
    "Rules:\n"
    "1. Answer only using facts present in the context below.\n"
    "2. If the context does not contain the answer, say so explicitly - "
    "do not guess or use outside knowledge.\n"
    "3. Keep the answer short and direct (2-5 sentences unless the "
    "question needs a list).\n"
    "4. Do not fabricate numbers, dates, or scheme names."
)


def build_prompt(question: str, retrieved_chunks: list) -> str:
    context_blocks = []
    for r in retrieved_chunks:
        chunk = r["chunk"]
        context_blocks.append(f"[Source: {chunk['title']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )


def call_llm(question: str, retrieved_chunks: list) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Get a free API key from Google AI Studio "
            "(https://aistudio.google.com/app/apikey), add it to your .env file, "
            "and export it before running the CLI."
        )

    client = genai.Client(api_key=api_key)
    prompt = build_prompt(question, retrieved_chunks)

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=500,
                temperature=0.0,
            )
        )
    except Exception as e:
        raise RuntimeError(
            f"Google Gemini API error: {str(e)}"
        ) from e

    return response.text

"""
generate.py
-----------
Step 4 of the RAG pipeline: construct a grounded prompt from the
retrieved chunks and call an LLM to produce a short, source-grounded
answer.

LLM: Claude, via the Anthropic API (anthropic Python SDK).
Why Claude/Anthropic: a single, well-documented SDK, strong instruction
following for "answer only from context" constraints, and the assignment
allows any LLM. Swapping to OpenAI/Gemini/a local model only requires
rewriting `call_llm()` below - the prompt-construction logic is
provider-agnostic.

Requires an ANTHROPIC_API_KEY environment variable (see .env.example).
"""
import os

import anthropic

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

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
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env, "
            "fill in your key, and export it (or use a tool like "
            "python-dotenv / `source .env`) before running the CLI."
        )

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_prompt(question, retrieved_chunks)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError as e:
        raise RuntimeError(
            "Anthropic API rejected the key (invalid or expired). "
            "Check ANTHROPIC_API_KEY in your .env file."
        ) from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError(
            "Could not reach the Anthropic API. Check your internet connection."
        ) from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(
            f"Anthropic API returned an error (status {e.status_code}): {e.message}"
        ) from e

    return response.content[0].text

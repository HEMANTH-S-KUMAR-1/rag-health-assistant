"""
ingest.py
---------
Step 1 of the RAG pipeline: turn the PIB backgrounder page into clean,
meaning-aligned chunks of 200-500 words.

Design choice: chunk along the document's own markdown headers (##, ###)
rather than a fixed-size sliding window. The source document is already
organised into named sections (AB-PMJAY, AAM, PM-ABHIM, ABDM, individual
NHM programmes...), so header-aligned chunking keeps each chunk topically
coherent, which is exactly what the assignment asks for.

Usage:
    python src/ingest.py
    -> reads data/raw_page.md
    -> writes data/chunks.json
"""
import json
import re
from pathlib import Path

RAW_PAGE_PATH = Path(__file__).parent.parent / "data" / "raw_page.md"
CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks.json"

MIN_WORDS = 200
MAX_WORDS = 500


def word_count(text: str) -> int:
    return len(text.split())


def split_into_sections(raw_text: str):
    """
    Split the raw markdown on '##' / '###' headers.
    Returns a list of (title, body) tuples, in document order.
    The very first block (before any header) becomes a "Preamble" section.
    """
    lines = raw_text.splitlines()
    sections = []
    current_title = "Preamble"
    current_lines = []

    for line in lines:
        header_match = re.match(r"^(#{2,3})\s+(.*)", line)
        if header_match:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = header_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    # Drop the tiny metadata-only preamble block (source/title/date lines)
    sections = [(t, b) for t, b in sections if b and word_count(b) > 5]
    return sections


def split_long_section(title: str, body: str):
    """
    If a section exceeds MAX_WORDS, split it at paragraph boundaries into
    multiple sub-chunks, each kept as close to the 200-500 word band as
    possible without splitting a paragraph in half.
    """
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks = []
    buf = []
    buf_words = 0

    for para in paragraphs:
        para_words = word_count(para)
        if buf and buf_words + para_words > MAX_WORDS:
            chunks.append("\n\n".join(buf))
            buf = [para]
            buf_words = para_words
        else:
            buf.append(para)
            buf_words += para_words

    if buf:
        chunks.append("\n\n".join(buf))

    # Merge a trailing too-small chunk into the previous one, but only if
    # that doesn't push the combined chunk back over MAX_WORDS - an
    # undersized final chunk is preferable to an oversized one.
    if len(chunks) > 1 and word_count(chunks[-1]) < MIN_WORDS:
        combined_words = word_count(chunks[-2]) + word_count(chunks[-1])
        if combined_words <= MAX_WORDS:
            chunks[-2] = chunks[-2] + "\n\n" + chunks[-1]
            chunks.pop()

    return chunks


def merge_short_sections(sections):
    """
    Merge any section under MIN_WORDS into the following section so no
    chunk is left too small to carry standalone meaning (e.g. a one-line
    sub-heading with a single stat).
    """
    merged = []
    carry_title, carry_body = None, ""

    for title, body in sections:
        if carry_body:
            body = carry_body + "\n\n" + body
            title = f"{carry_title} / {title}"  # keep both topics visible
            carry_title, carry_body = None, ""

        if word_count(body) < MIN_WORDS:
            carry_title, carry_body = title, body
            continue

        merged.append((title, body))

    if carry_body:  # trailing leftover
        if merged:
            last_title, last_body = merged[-1]
            merged[-1] = (last_title, last_body + "\n\n" + carry_body)
        else:
            merged.append((carry_title, carry_body))

    return merged


def build_chunks():
    raw_text = RAW_PAGE_PATH.read_text(encoding="utf-8")
    sections = split_into_sections(raw_text)
    sections = merge_short_sections(sections)

    chunks = []
    chunk_id = 0
    for title, body in sections:
        pieces = split_long_section(title, body) if word_count(body) > MAX_WORDS else [body]
        for i, piece in enumerate(pieces):
            chunk_title = title if len(pieces) == 1 else f"{title} (part {i + 1})"
            chunks.append(
                {
                    "id": chunk_id,
                    "title": chunk_title,
                    "text": piece,
                    "word_count": word_count(piece),
                }
            )
            chunk_id += 1

    return chunks


def main():
    chunks = build_chunks()
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    word_counts = [c["word_count"] for c in chunks]
    print(f"Wrote {len(chunks)} chunks to {CHUNKS_PATH}")
    print(f"Word count range: {min(word_counts)}-{max(word_counts)}, "
          f"avg {sum(word_counts) / len(word_counts):.0f}")
    for c in chunks:
        flag = "  <-- outside 200-500 band" if not (MIN_WORDS <= c["word_count"] <= MAX_WORDS) else ""
        print(f"  [{c['id']:>2}] {c['word_count']:>4}w  {c['title']}{flag}")


if __name__ == "__main__":
    main()

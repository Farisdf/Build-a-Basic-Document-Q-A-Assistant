"""
Basic Document Q&A Assistant
============================

A small retrieval-augmented generation (RAG) pipeline over local text documents.

Pipeline (see ``ask``):
    load docs -> chunk -> embed (all-MiniLM-L6-v2) -> cosine-similarity retrieval (top 4)
    -> build a citation-labelled prompt -> call Claude -> validate strict-JSON output with
    Pydantic (one retry on failure) -> print answer + citations + retrieved chunks.

The API key is read ONLY from the ``ANTHROPIC_API_KEY`` environment variable (optionally
loaded from a local ``.env`` file via python-dotenv). It is never hard-coded.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).parent / "docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "claude-sonnet-4-6"
TOP_K = 4
MAX_CHUNK_WORDS = 200  # target upper bound for a chunk; paragraphs are merged up to this

# Phrases that count as an honest "I don't know" from the model.
DONT_KNOW_PATTERNS = (
    "i don't know",
    "i do not know",
    "don't have enough information",
    "do not have enough information",
    "not contain",  # "the passages do not contain ..."
    "no information",
)

# ---------------------------------------------------------------------------
# Step 8 — Pydantic model for the LLM's strict JSON output
# ---------------------------------------------------------------------------


class QAResponse(BaseModel):
    """The exact JSON shape the model is asked to return."""

    answer: str
    citations: list[int]


# ---------------------------------------------------------------------------
# Step 2 — Chunking
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A passage of text plus the metadata needed to trace it back to its source."""

    source: str  # file name inside docs/
    chunk_index: int  # 0-based index of this chunk within its source file
    text: str

    @property
    def label(self) -> str:
        return f"{self.source}#chunk{self.chunk_index}"


def _split_long_paragraph(paragraph: str, max_words: int) -> list[str]:
    """Split a single over-long paragraph into pieces of at most ``max_words`` words."""
    words = paragraph.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def chunk_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """
    Split ``text`` into passages.

    Strategy: split on blank lines into paragraphs (markdown headings are attached to the
    paragraph that follows them so the chunk keeps its topic), then greedily merge
    consecutive paragraphs while the running word count stays under ``max_words``.
    Any single paragraph longer than ``max_words`` is hard-split by word count.
    """
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # Attach headings ("# ..." / "## ...") to the following paragraph.
    paragraphs: list[str] = []
    pending_heading: str | None = None
    for para in raw_paragraphs:
        if re.match(r"^#{1,6}\s", para) and "\n" not in para:
            pending_heading = para.lstrip("#").strip()
            continue
        if pending_heading:
            para = f"{pending_heading}\n{para}"
            pending_heading = None
        paragraphs.append(para)
    if pending_heading:  # heading with no body at the very end
        paragraphs.append(pending_heading)

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_words = 0
    for para in paragraphs:
        n = len(para.split())
        if n > max_words:
            if buffer:
                chunks.append("\n\n".join(buffer))
                buffer, buffer_words = [], 0
            chunks.extend(_split_long_paragraph(para, max_words))
            continue
        if buffer and buffer_words + n > max_words:
            chunks.append("\n\n".join(buffer))
            buffer, buffer_words = [], 0
        buffer.append(para)
        buffer_words += n
    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def load_and_chunk_documents(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Load every .txt / .md file in ``docs_dir`` and return its chunks with metadata."""
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")
    files = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in {".txt", ".md"})
    if not files:
        raise FileNotFoundError(f"No .txt or .md files found in {docs_dir}")

    chunks: list[Chunk] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for idx, passage in enumerate(chunk_text(text)):
            chunks.append(Chunk(source=path.name, chunk_index=idx, text=passage))
    return chunks


# ---------------------------------------------------------------------------
# Step 3 & 4 — Embeddings cached in memory + cosine-similarity retrieval
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity to the question


@dataclass
class KnowledgeBase:
    """Holds all chunks and their embeddings (a single in-memory numpy array)."""

    chunks: list[Chunk]
    embeddings: np.ndarray = field(repr=False)  # shape (n_chunks, dim), L2-normalised
    _model: Any = field(default=None, repr=False)

    @classmethod
    def build(cls, docs_dir: Path = DOCS_DIR, model_name: str = EMBEDDING_MODEL_NAME) -> "KnowledgeBase":
        # Imported lazily so that `--help` and chunking-only usage stay fast.
        from sentence_transformers import SentenceTransformer

        chunks = load_and_chunk_documents(docs_dir)
        model = SentenceTransformer(model_name)
        # normalize_embeddings=True -> dot product == cosine similarity
        embeddings = model.encode(
            [c.text for c in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return cls(chunks=chunks, embeddings=embeddings.astype(np.float32), _model=model)

    def embed_question(self, question: str) -> np.ndarray:
        vec = self._model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        return vec[0].astype(np.float32)

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks most similar (cosine) to ``question``, best first."""
        q = self.embed_question(question)
        # Both sides are unit vectors, so the dot product is the cosine similarity.
        scores = self.embeddings @ q
        top_idx = np.argsort(-scores)[:top_k]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Step 5 — Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a careful question-answering assistant for a document collection.

Rules you must follow:
1. Answer ONLY using the numbered passages supplied in the user message. Do not use any
   outside knowledge, and do not guess.
2. If the passages do not contain the information needed to answer the question, set
   "answer" to exactly "I don't know" and set "citations" to an empty list []. Saying
   "I don't know" is the correct and expected behaviour in that case - never invent an answer.
3. When you do answer, cite the passage number(s) you relied on in "citations" using the
   integer labels shown in square brackets (for example [2] -> 2).
4. Reply with STRICT JSON only - no markdown, no code fences, no commentary before or after.
   The JSON must match this shape exactly:
{"answer": "string", "citations": [1, 3]}
"""


def build_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    """Assemble the user message: numbered passages + the question + format reminder."""
    passage_blocks = []
    for i, r in enumerate(retrieved, start=1):
        passage_blocks.append(f"[{i}] (source: {r.chunk.label})\n{r.chunk.text}")
    passages = "\n\n".join(passage_blocks)
    return (
        "Here are the retrieved passages:\n\n"
        f"{passages}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above. If they do not contain the answer, reply with "
        '"I don\'t know". Respond with strict JSON of the form '
        '{"answer": "string", "citations": [1, 3]} and nothing else.'
    )


# ---------------------------------------------------------------------------
# Step 5 & 8 — LLM call with validation and one retry
# ---------------------------------------------------------------------------


def _get_client():
    """Create an Anthropic client. The key comes only from the environment."""
    import anthropic

    load_dotenv()  # loads .env into os.environ if the file exists; no-op otherwise
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Put it in a .env file (see .env.example) "
            "or export it in your shell before running."
        )
    return anthropic.Anthropic(api_key=api_key)


def _extract_json(raw: str) -> str:
    """Be tolerant of code fences or stray prose around the JSON object."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text


def parse_response(raw: str, n_passages: int) -> QAResponse:
    """Parse + validate the model's text into ``QAResponse``. Raises on any failure."""
    data = json.loads(_extract_json(raw))
    resp = QAResponse.model_validate(data)
    bad = [c for c in resp.citations if c < 1 or c > n_passages]
    if bad:
        raise ValueError(f"citation numbers out of range 1..{n_passages}: {bad}")
    return resp


def _message_text(message) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


def call_llm(prompt: str, n_passages: int, client=None) -> tuple[QAResponse | None, str, bool]:
    """
    Call Claude with ``prompt``; parse the strict-JSON reply into ``QAResponse``.

    On a parse/validation failure the call is retried ONCE with a clarifying follow-up
    instruction. Returns ``(parsed_or_None, last_raw_text, validation_passed)``.
    """
    import anthropic

    client = client or _get_client()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    raw = ""

    for attempt in (1, 2):
        try:
            message = client.messages.create(
                model=LLM_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except anthropic.RateLimitError as e:
            print(f"[error] Rate limited by the API: {e}", file=sys.stderr)
            return None, raw, False
        except anthropic.APIStatusError as e:
            print(f"[error] API returned status {e.status_code}: {e.message}", file=sys.stderr)
            return None, raw, False
        except anthropic.APIConnectionError as e:
            print(f"[error] Could not reach the API: {e}", file=sys.stderr)
            return None, raw, False

        if message.stop_reason == "refusal":
            print("[error] The model declined to answer this request.", file=sys.stderr)
            return None, raw, False

        raw = _message_text(message)
        try:
            return parse_response(raw, n_passages), raw, True
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            print(f"[warn] attempt {attempt}: response failed validation ({e}).", file=sys.stderr)
            if attempt == 1:
                # Feed the bad reply back and ask for a corrected, format-compliant answer.
                messages.append({"role": "assistant", "content": raw or "(empty)"})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid. It must be a single strict JSON "
                            'object of the form {"answer": "string", "citations": [1, 3]} with '
                            "no code fences and no extra text. \"citations\" must be a list of "
                            f"integers between 1 and {n_passages} (or [] if you don't know). "
                            "Please resend the corrected JSON only."
                        ),
                    }
                )

    return None, raw, False


# ---------------------------------------------------------------------------
# Step 6 — "I don't know" detection (used for reporting / tests only; the model is free
# to say it and the code simply reports that honestly)
# ---------------------------------------------------------------------------


def is_dont_know(answer: str) -> bool:
    a = answer.strip().lower()
    return any(p in a for p in DONT_KNOW_PATTERNS)


# ---------------------------------------------------------------------------
# Step 9 — Full pipeline
# ---------------------------------------------------------------------------


@dataclass
class AskResult:
    question: str
    retrieved: list[RetrievedChunk]
    response: QAResponse | None
    raw_llm_text: str
    validation_passed: bool

    @property
    def answer(self) -> str:
        return self.response.answer if self.response else "(no valid answer - see error above)"

    @property
    def citations(self) -> list[int]:
        return self.response.citations if self.response else []

    @property
    def dont_know(self) -> bool:
        return bool(self.response) and is_dont_know(self.response.answer)

    def cited_chunks(self) -> list[tuple[int, RetrievedChunk]]:
        """Map citation numbers back to the actual chunks (1-based -> retrieved list)."""
        return [(n, self.retrieved[n - 1]) for n in self.citations if 1 <= n <= len(self.retrieved)]


def ask(question: str, kb: KnowledgeBase, client=None, verbose: bool = True) -> AskResult:
    """
    Run the whole pipeline for one question:
    embed -> retrieve top-4 -> build prompt -> call LLM -> validate (1 retry) -> report.
    """
    retrieved = kb.retrieve(question, top_k=TOP_K)
    prompt = build_prompt(question, retrieved)
    response, raw, ok = call_llm(prompt, n_passages=len(retrieved), client=client)
    result = AskResult(
        question=question, retrieved=retrieved, response=response, raw_llm_text=raw, validation_passed=ok
    )
    if verbose:
        print(format_result(result))
    return result


def _snippet(text: str, n: int = 110) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[: n - 3] + "..."


def format_result(r: AskResult) -> str:
    lines = [
        "=" * 78,
        f"Q: {r.question}",
        "-" * 78,
        "Retrieved chunks (top 4 by cosine similarity):",
    ]
    for i, rc in enumerate(r.retrieved, start=1):
        lines.append(f"  [{i}] {rc.chunk.label}  (score={rc.score:.3f})")
        lines.append(f"      {_snippet(rc.chunk.text)}")
    lines.append("-" * 78)
    if r.response is None:
        lines.append("ANSWER: could not obtain a valid response from the model after 1 retry.")
        if r.raw_llm_text:
            lines.append(f"Last raw model output: {r.raw_llm_text!r}")
    else:
        lines.append(f"ANSWER: {r.answer}")
        if r.dont_know:
            lines.append("  (the documents do not cover this - the assistant declined to guess)")
        lines.append(f"CITATIONS: {r.citations if r.citations else 'none'}")
        for n, rc in r.cited_chunks():
            lines.append(f"  [{n}] -> {rc.chunk.label}: {_snippet(rc.chunk.text, 90)}")
    lines.append(f"VALIDATION: {'PASSED' if r.validation_passed else 'FAILED'}")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Ask questions about the documents in docs/.")
    parser.add_argument("question", nargs="?", help="A single question. Omit for interactive mode.")
    parser.add_argument("--docs", type=Path, default=DOCS_DIR, help="Path to the docs folder.")
    args = parser.parse_args(argv)

    print(f"Loading and embedding documents from {args.docs} ...", file=sys.stderr)
    kb = KnowledgeBase.build(args.docs)
    print(f"Indexed {len(kb.chunks)} chunks from {len({c.source for c in kb.chunks})} files.\n", file=sys.stderr)

    try:
        client = _get_client()
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    if args.question:
        ask(args.question, kb, client=client)
        return 0

    print("Interactive mode. Type a question, or 'quit' to exit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"quit", "exit", "q"}:
            break
        ask(q, kb, client=client)
    return 0


if __name__ == "__main__":
    sys.exit(main())

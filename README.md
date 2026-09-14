# Basic Document Q&A Assistant

A small, dependency-light retrieval-augmented generation (RAG) assistant that answers
questions about a local folder of text documents. It only answers from what the documents
say, cites the passages it used, and says **"I don't know"** when the documents don't
contain the answer.

The bundled knowledge base is a fictional university course, **CS 201: Data Structures and
Algorithms** (syllabus, grading policy, exam/lab rules, assignment rules). Swap the files in
`docs/` for your own `.txt` / `.md` files and everything else works unchanged.

## How it works

```
docs/*.md  ──►  chunk (by paragraph, ≤ ~200 words)  ──►  embed with all-MiniLM-L6-v2
                                                              │  (cached in a numpy array)
question  ──►  embed  ──►  cosine similarity  ──►  top-4 chunks
                                                              │
                     prompt = numbered passages [1]..[4] + question + "answer only from passages,
                              say I don't know otherwise, reply in strict JSON"
                                                              │
                     Gemini (gemini-3.6-flash)  ──►  {"answer": "...", "citations": [1, 3]}
                                                              │
                     Pydantic QAResponse validation  ──►  retry once on failure  ──►  print
```

Key points:

- **Chunking** – every file in `docs/` is split into passages (markdown headings are kept
  with the paragraph that follows; paragraphs are merged up to ~200 words). Each chunk
  carries its source filename and chunk index.
- **Embeddings** – `sentence-transformers` / `all-MiniLM-L6-v2`; all chunk vectors are
  L2-normalised and kept in one in-memory numpy array. No vector database.
- **Retrieval** – the question is embedded with the same model; cosine similarity is a dot
  product against the array; the top 4 chunks are returned. Only those 4 chunks are sent to
  the LLM, never the whole corpus.
- **LLM call** – Google `google-genai` SDK, model `gemini-3.6-flash`. The API key is read only from
  the `GEMINI_API_KEY` environment variable (optionally loaded from `.env`).
- **Strict JSON + validation** – the model is told to return
  `{"answer": "string", "citations": [1, 3]}`. The reply is parsed into a Pydantic
  `QAResponse`. If parsing or validation fails (bad JSON, wrong types, citation numbers
  outside 1..4) the call is retried **once** with a clarifying follow-up message. If it
  still fails, a clear error is printed and the program keeps running.
- **"I don't know"** – the prompt explicitly allows and expects it. The code never forces a
  confident answer; it just reports what the model said and flags it.
- **Citations** – printed answers show the citation numbers, and each number is mapped back to
  `filename#chunkN` plus a snippet of the chunk text so every citation is verifiable.

## Folder structure

```
doc-qa-assistant/
├── docs/                      # knowledge base (4 markdown files about CS 201)
│   ├── course_syllabus.md
│   ├── grading_policy.md
│   ├── exam_and_lab_rules.md
│   └── assignments_and_tools.md
├── qa_assistant.py            # the full pipeline + CLI (ask(), KnowledgeBase, QAResponse ...)
├── test_qa.py                 # runs the 5 required test questions, writes TEST_RESULTS.md
├── TEST_RESULTS.md            # output of the last test run
├── requirements.txt
├── .env.example               # shows the required env var name (no real key)
├── .gitignore                 # ignores .env
└── README.md
```

## Installation

Python 3.10+ is required.

```bash
cd doc-qa-assistant
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the `all-MiniLM-L6-v2` model (~90 MB) from Hugging Face and caches it.

## Setting the API key

The assistant reads `GEMINI_API_KEY` from the environment and nowhere else. Two options:

**Option A – `.env` file (loaded automatically by python-dotenv)**

```bash
cp .env.example .env
# then edit .env and put your real key after GEMINI_API_KEY=
```

`.env` is listed in `.gitignore`, so it is never committed.

**Option B – export it in your shell**

```bash
# macOS / Linux
export GEMINI_API_KEY=AIza...

# Windows PowerShell
$env:GEMINI_API_KEY = "AIza..."

# Windows cmd
set GEMINI_API_KEY=AIza...
```

## Running the assistant

Ask a single question:

```bash
python qa_assistant.py "What percentage of the final grade is the final exam worth?"
```

Or start an interactive session (type `quit` to exit):

```bash
python qa_assistant.py
```

Example output:

```
==============================================================================
Q: What percentage of the final grade is the final exam worth?
------------------------------------------------------------------------------
Retrieved chunks (top 4 by cosine similarity):
  [1] grading_policy.md#chunk0  (score=0.641)
      Grade breakdown The final course grade is made up of four components: ...
  [2] grading_policy.md#chunk1  (score=0.445)
      ...
------------------------------------------------------------------------------
ANSWER: The final exam is worth 30% of the final grade.
CITATIONS: [1]
  [1] -> grading_policy.md#chunk0: Grade breakdown The final course grade ...
VALIDATION: PASSED
==============================================================================
```

To use a different document folder: `python qa_assistant.py --docs path/to/folder "question"`.

## Running the tests

```bash
python test_qa.py
```

This runs the five required questions (three direct, one re-worded/synonym question, one
that is not covered by the docs), prints the retrieved chunks, answer, citations and
validation status for each, checks that the out-of-scope question yields "I don't know",
and writes the full transcript to `TEST_RESULTS.md`. The exit code is 0 only if every
behavioural check passes.

## Using it as a library

```python
from qa_assistant import KnowledgeBase, ask

kb = KnowledgeBase.build()                 # loads docs/, chunks, embeds once
result = ask("When are the office hours?", kb)
print(result.answer, result.citations)
for n, chunk in result.cited_chunks():     # map citation numbers back to sources
    print(n, chunk.chunk.source, chunk.chunk.chunk_index)
```

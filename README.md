# 📖 Book Forge

**Any text in, a publish-ready book out — 100% local, via [Ollama](https://ollama.com).**

Book Forge turns manuscripts, notes, drafts or PDFs into a structured book with a title, preface,
numbered chapters and a glossary, then exports it as **PDF**, **EPUB** or **Markdown** — all without
sending a single word to the cloud.

---

## Table of contents

- [Why Book Forge](#why-book-forge)
- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the app](#running-the-app)
- [Using the app](#using-the-app)
- [Exports](#exports)
- [Configuration](#configuration)
- [Supported inputs](#supported-inputs)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Privacy](#privacy)
- [Development](#development)
- [License](#license)

---

## Why Book Forge

Most "AI book writers" invent content. Book Forge is built around the opposite promise: **your text,
polished — not replaced.** Every rewritten passage is checked against the source with embeddings, and
borderline cases are verified by a decision model. If a passage can't be rewritten faithfully, the
original is kept and you are told about it.

- **Local by default** — runs against Ollama on your own machine.
- **Fidelity-gated rewriting** — cosine similarity plus a decision-model check per passage.
- **Editable everywhere** — title, preface, chapters and glossary can all be edited by hand.
- **Undo built in** — every edit is snapshotted, so a bad assistant action is one click away from gone.

## Features

| Area | What you get |
| --- | --- |
| **Create** | Upload PDF / DOCX / TXT / MD or paste raw text. A decision model infers genre, tone, voice, audience, chapter count, and whether the book needs a preface or glossary. |
| **Polish** | The source is split into ~900-word passages and rewritten one by one in the author's own voice, with the preceding text as context so transitions stay smooth. |
| **Verify** | Each passage is embedded and compared with the original. Similarity below `0.75` triggers a decision-model yes/no check; failures fall back to the original text. |
| **Preview** | Read the finished book (title page, contents, preface, chapters, glossary) or the print layout with the PDF viewer embedded in the app. |
| **Edit** | Edit the title, author, preface, every chapter and the glossary. Add, delete and reorder chapters. Undo or revert to the originally generated book. |
| **Assistant** | A context-aware chat with its own model, memory and retrieval over the book. It can answer questions about the text or apply edits: rewrite a chapter, find & replace, rename, add, delete, move chapters, update the glossary, regenerate the preface. |
| **Projects** | Save the book and assistant memory as a `.bookforge.json` file and reopen it later. |
| **Exports** | Styled PDF (ReportLab), EPUB with a table of contents (EbookLib), Markdown, and the project file. |

## How it works

```
upload / paste
      │
      ▼
extract text  (pypdf · python-docx · plain text)
      │
      ▼
decision model ──► blueprint: genre, tone, voice, audience, chapters, preface, glossary
      │
      ▼
embed + segment ──► chapter boundaries
      │
      ▼
polish each ~900-word passage (parallel) ──► embedding check ──► decision-model check
      │                                            │
      │                                            └── below threshold? keep the original
      ▼
chapter titles + summaries ──► preface ──► glossary
      │
      ▼
preview · edit · assistant · export (PDF / EPUB / Markdown / project)
```

## Requirements

- **Python 3.10+**
- **[Ollama](https://ollama.com/download) 0.35 or newer** running locally
  (0.35+ is needed for the `/v1/systemone` decision endpoint)
- Enough RAM/VRAM for your chosen models. A quantized 20B writer model needs roughly 16 GB of
  memory; smaller models work fine and are much faster.

### Models

Book Forge uses three roles. The sidebar lets you pick any installed model for each one; the app
pre-selects these defaults when they are installed:

| Role | Default preference | Purpose |
| --- | --- | --- |
| **Decision** | `nimble:9b`, `nimble` | Structured choices and yes/no faithfulness checks (never writes prose) |
| **Writer** | `gpt-oss:20b`, `gemma4:12b` | Titling, polishing, chapter titles, preface and glossary |
| **Embedding** | `mimic`, `nomic-embed-text` | Chunking, retrieval for the assistant, fidelity scoring |

If none of the preferred names are installed, the first installed model is used, so make sure the
roles you pick actually fit the job (a decision model for decisions, an embedding model for embeddings).

Pull the defaults with:

```bash
ollama pull nimble:9b
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

> Model names and availability change. Check `ollama list` and pick whatever your Ollama install
> provides; the app never hard-codes a model you don't have.

## Installation

```bash
git clone https://github.com/<your-user>/bookforge.git
cd bookforge

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Make sure Ollama is running:

```bash
ollama serve
```

## Running the app

```bash
streamlit run app.py
```

Streamlit opens <http://localhost:8501>. Choose your models in the sidebar, upload or paste your
manuscript, and press **Publish my book**.

## Using the app

1. **Pick models** in the sidebar. Optionally set the book title and author; leave the title blank to
   let the model decide.
2. **Upload or paste** your source in *Create or open a book*. Supported: `.pdf`, `.docx`, `.txt`,
   `.md` (legacy `.doc` must be re-saved as `.docx`).
3. **Wait for the polish pass.** Progress, elapsed time and an ETA are shown per passage. On a GPU,
   raise *Parallel passages* and start Ollama with a matching `OLLAMA_NUM_PARALLEL`.
4. **Review** the result in *Preview*. A fidelity score is shown; passages that could not be
   rewritten safely are reported and kept as your original text.
5. **Edit** in the *Edit* tab — manually or through the *Assistant*. Every change is undoable.
6. **Export** from the *Export* tab, or save the project file to continue later.

### Assistant examples

- "Which chapter introduces the main character?"
- "Make chapter 3 more concise without losing any facts."
- "Replace every occurrence of 'utilize' with 'use'."
- "Move the glossary chapter before the appendix."
- "Add a chapter about deployment."
- "Regenerate the preface now that the chapters changed."

The assistant plans structured operations (`rewrite`, `find_replace`, `rename_chapter`,
`add_chapter`, `delete_chapter`, `move_chapter`, `add_term`, …), applies them to the book, and
reports what it did.

## Exports

| Format | Details |
| --- | --- |
| **PDF** | 6 × 9 inch print layout with a title page, table of contents, PDF bookmarks and page numbers. Tables and code blocks are rendered; if ReportLab cannot lay out a passage, the export retries with tables flattened so one awkward page can't block the book. |
| **EPUB** | Per-section XHTML, embedded CSS, navigation and NCX table of contents, correct language metadata. |
| **Markdown** | The full book as a single `.md` file. |
| **Project** | `.bookforge.json` containing the book and assistant memory, re-openable from *Create or open a book*. |

## Configuration

Book Forge reads standard Ollama environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_HOST` | `127.0.0.1:11434` | Where Ollama listens. Use `http://host:port` for a remote instance. |
| `OLLAMA_NUM_PARALLEL` | Ollama default | Set this on the Ollama server to match *Parallel passages* for real speed-ups. |

The UI theme and upload limit live in [`.streamlit/config.toml`](.streamlit/config.toml). Context
length and parallelism are adjustable in the sidebar; the assistant has its own context setting in
its tab.

## Supported inputs

- `.pdf` — text extraction via `pypdf`, with de-hyphenation and paragraph re-flow. Scanned/image-only
  PDFs have no text layer and won't work without OCR first.
- `.docx` — paragraphs, heading styles and tables via `python-docx`.
- `.txt`, `.md` — read as UTF-8 (invalid bytes are replaced, not fatal).
- Pasted text of any length.

Very short inputs (under 5 words) are rejected.

## Project structure

```
.
├── app.py                     # The entire application: pipeline, UI, exports
├── requirements.txt           # Runtime dependencies
├── ruff.toml                  # Lint configuration
├── .gitattributes             # Consistent line endings across platforms
├── .streamlit/
│   └── config.toml            # Theme, upload limit, telemetry off
├── .github/
│   └── workflows/ci.yml       # Lint + syntax check on push / PR
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
└── README.md
```

Generated books, exports and `.bookforge.json` projects are git-ignored — they contain your content.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "Can't reach Ollama" | Start it with `ollama serve`, confirm `ollama list` works, then refresh the page. |
| Decision call fails | You need Ollama 0.35+ for `/v1/systemone`, and a decision-capable model (e.g. `nimble`) selected. |
| Rewrites are slow | Lower *Parallel passages*, use a smaller writer model, or run Ollama with `OLLAMA_NUM_PARALLEL` matching the slider. |
| Out of memory | Reduce the context length, use a smaller/quantized model, or lower parallelism. |
| Passages "could not be rewritten" | The fidelity gate rejected them and the original text was kept. Try a stronger writer model or a larger context. |
| PDF export fails | The app automatically retries with plain tables. If it still fails, the error dialog shows a traceback you can open an issue with. |
| `.doc` upload rejected | Save the file as `.docx` first; legacy binary Word files are not supported. |
| Assistant ignores an instruction | Give it a specific target ("chapter 2", "the glossary") and check the model selected in the Assistant tab. |

## Privacy

Everything runs locally: documents are parsed in-process, model calls go to your Ollama instance,
and exports are generated in memory. No analytics, no external APIs. Streamlit telemetry is disabled
in `.streamlit/config.toml`. If you point `OLLAMA_HOST` at a remote server, your text leaves your
machine — that is your choice, not the default.

## Development

```bash
pip install ruff
ruff check .
python -m compileall -q app.py
```

CI runs the same two commands on every push and pull request. There is no automated test suite yet;
contributions that add tests for text extraction, fidelity gating or export building are welcome.

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, style and pull-request guidelines.

## License

[MIT](LICENSE) © Book Forge contributors

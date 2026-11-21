# Changelog

All notable changes to Book Forge are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-11-21

### Added

- Local book generation from PDF, DOCX, TXT, Markdown or pasted text via Ollama.
- Decision-model blueprint: genre, tone, voice, audience, chapter count, preface and glossary.
- Passage-level polishing with embedding-based fidelity scoring and a decision-model check for
  borderline rewrites; unfaithful passages fall back to the original text.
- Preview tab with the full book and an embedded PDF print layout.
- Edit tab for title, author, preface, chapters and glossary, with add/delete/reorder, undo and
  revert-all.
- Context-aware assistant with its own model, memory and retrieval, able to apply structured edits.
- Exports: 6 × 9 inch PDF, EPUB with navigation, Markdown and a re-openable `.bookforge.json`
  project file.
- Sidebar controls for model selection, context length and parallel passage processing.

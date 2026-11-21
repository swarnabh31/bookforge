# Contributing to Book Forge

Thanks for wanting to improve Book Forge. This project is a single-file Streamlit app, so the setup
is deliberately small.

## Getting set up

```bash
git clone https://github.com/swarnabh31/bookforge.git
cd bookforge
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
pip install ruff
```

You also need [Ollama](https://ollama.com) 0.35+ running locally with a decision model, a writer
model and an embedding model installed. See the [README](README.md#requirements) for details.

## Before opening a pull request

Run the same checks CI runs:

```bash
ruff check .
python -m compileall -q app.py
```

Then start the app and exercise the path you touched:

```bash
streamlit run app.py
```

For changes to the pipeline, a small upload plus a short paste is usually enough. For export changes,
verify all four outputs (PDF, EPUB, Markdown, project file).

## What to keep in mind

- **Stay local-first.** No telemetry, no external APIs, no required cloud keys.
- **Don't bypass the fidelity gate.** Rewrites must remain verifiable against the source. If a model
  cannot rewrite a passage faithfully, the original text wins.
- **Keep it a single file.** `app.py` is intentionally self-contained; prefer adding a function over
  adding a module unless there is a strong reason.
- **Match the existing style.** Type hints where they help, Pydantic for structured model output,
  no comments that restate the code.
- **Never commit user content or secrets.** `.bookforge.json` projects, exports and
  `.streamlit/secrets.toml` are git-ignored — keep it that way.
- **New dependencies** need a good reason and a line in `requirements.txt`.

## Reporting bugs

Open an issue with:

1. What you did (input type, model choices, steps).
2. What you expected and what happened.
3. The full traceback from the app, if there is one.
4. Your OS, Python version, `ollama --version` and `ollama list` output.

Please don't attach manuscripts you can't share publicly — redact or reproduce with a short sample.

## Suggesting features

Open an issue describing the workflow you want, not just the UI control. Book Forge optimises for
faithful, local book production, so features that trade fidelity or privacy for convenience are
unlikely to be accepted.

## License

By contributing you agree that your contributions are licensed under the [MIT License](LICENSE).

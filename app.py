"""Book Forge: any text in, a publish-ready book out (PDF / EPUB). 100% local via Ollama.

New in this version
  * Preview   : read the whole book (title page, contents, preface, chapters, glossary) or the print layout (PDF).
  * Edit      : edit title, author, preface, every chapter and the glossary; add / delete / reorder chapters; undo.
  * Assistant : context-aware chat bot with its own memory and its own model dropdown. It answers questions about the
                book (retrieval over the text) and can edit it (rewrite, find/replace, rename, add, delete, move...).
  * Projects  : save the book + assistant memory as a .json file and re-open it later.

Run:  streamlit run app.py
"""
import base64
import hashlib
import html
import io
import json
import os
import re
import time
import traceback
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import httpx
import numpy as np
import ollama
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

client = ollama.Client()
EMB_CTX = {"num_ctx": 8192}


# Context windows are user-settable (sidebar / assistant tab). Plain module state because the polishing pass runs in threads.
SETTINGS = {"ctx": 8192, "chat_ctx": 32768, "override": 0}


def ctx_for(model):  # gpt-oss needs room for its hidden reasoning, so it never gets less than 16k
    n = SETTINGS["override"] or SETTINGS["ctx"]
    return {"num_ctx": max(n, 16384) if model.startswith("gpt-oss") else n}


@contextmanager
def use_ctx(n: int):
    """Run LLM calls (chat answers, rewrites, new chapters...) with a bigger context window."""
    old, SETTINGS["override"] = SETTINGS["override"], n
    try:
        yield
    finally:
        SETTINGS["override"] = old


MIN_SIM, CHUNK_WORDS = 0.75, 900  # fidelity floor (cosine) and words per polishing pass


# ───────────────────────── Schemas (Pydantic = structure + formatting) ─────────────────────────
class Blueprint(BaseModel):
    title: str
    subtitle: str = ""
    genre: str = Field(description="e.g. non-fiction, fiction, tutorial, self-help, technical guide, memoir, children's")
    subject: str = Field(description="one sentence: what the text is about")
    tone: str = Field(description="e.g. warm and conversational, formal and academic")
    voice: str = Field(description="point of view / persona, e.g. first person singular")
    audience: str
    language: str = Field(default="en", description="ISO 639-1 code of the manuscript language")
    chapter_count: int = 1
    include_preface: bool = False
    include_glossary: bool = False
    numbering: str = "none"  # keep = author's own section numbers, auto = book numbers 1.1, 1.2…, none


class Titling(BaseModel):
    title: str
    subtitle: str = ""
    subject: str = Field(description="one sentence: what the text is about")
    language: str = Field(default="en", description="ISO 639-1 code of the manuscript language")


class ChapterMeta(BaseModel):
    title: str
    summary: str


class Term(BaseModel):
    term: str
    definition: str


class Glossary(BaseModel):
    terms: list[Term]


class Chapter(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])  # stable key for editor widgets
    title: str
    summary: str = ""
    body: str


NUMBERED = {"non-fiction", "self-help", "tutorial", "technical guide", "business", "academic"}  # genres with 1.1, 1.2…


CAPTION = re.compile(r"^(figure|fig\.|table|algorithm|listing|box|chart|appendix figure)\s*[\dA-Z]", re.I)
FIRST_PERSON = re.compile(r"^(I|I'll|I'm|I've|Let's|Let us)\s")


def tidy_headings(md: str) -> str:
    """Figure/table captions and whole sentences that PDF extraction turned into headings become plain bold lines."""
    out, fence = [], False
    for line in md.split("\n"):
        if line.lstrip().startswith("```"):
            fence = not fence
        elif not fence and (m := re.match(r"#{2,6}\s+(.*\S)\s*$", line)):
            core = re.sub(r"^\*\*|\*\*$", "", strip_numbers(line, True).lstrip("#").strip())
            if (CAPTION.match(core) or FIRST_PERSON.match(core) or len(core.split()) > 14
                    or (core.endswith(".") and len(core.split()) > 4)):
                line = f"\n**{m[1].strip()}**\n"
        out.append(line)
    return "\n".join(out)


def renumber(md: str, ch: int, mode: str) -> str:
    """Display-time numbering of headings (never changes the stored text).
    auto = '## ' headings become <chapter>.<n>; none = numbers removed; keep = untouched."""
    if mode == "keep":
        return md
    out, k, fence = [], 0, False
    for line in md.split("\n"):
        if line.lstrip().startswith("```"):
            fence = not fence
        elif not fence and re.match(r"#{2,6}\s", line):
            if mode == "auto" and line.startswith("## "):
                k += 1
                line = f"## {ch}.{k} {strip_numbers(line, True)[3:]}"
            else:
                line = strip_numbers(line, True)
        out.append(line)
    return "\n".join(out)


class Book(BaseModel):
    blueprint: Blueprint
    author: str = ""
    preface: str = ""
    chapters: list[Chapter]
    glossary: list[Term] = []
    fidelity: float = 0.0
    unpolished: int = 0  # passages kept as-is because the writer model failed on them
    why: str = ""  # reason for the first such failure

    def sections(self) -> list[tuple[str, str]]:
        s = [("Preface", self.preface)] if self.preface.strip() else []
        mode = self.blueprint.numbering
        s += [(f"Chapter {i}: {c.title}", renumber(tidy_headings(c.body), i, mode))
              for i, c in enumerate(self.chapters, 1)]
        if self.glossary:
            s.append(("Glossary", "\n\n".join(f"**{t.term}**: {t.definition}" for t in self.glossary)))
        return s


# ───────────────────────── Prompts ─────────────────────────
# nimble is a decision model: it only answers choice / yes-no / score questions (POST /v1/systemone), never free text.
def _q(instructions, options):
    return {"type": "choice", "instructions": instructions, "criteria": options}


QUESTIONS = {
    "genre": _q("Which genre best fits this manuscript?", {
        "fiction": "Novel or short stories with characters and plot", "children's story": "Story for young children",
        "non-fiction": "General non-fiction explaining ideas or facts", "memoir": "Personal life story or reflections",
        "self-help": "Personal growth, habits, motivation", "tutorial": "Step-by-step learning material",
        "technical guide": "Programming, engineering or technical documentation", "business": "Business, management, strategy",
        "academic": "Scholarly or research writing", "essays": "Opinion pieces, reflections, thoughts"}),
    "tone": _q("Which tone does the author's writing have?", {
        "warm and conversational": "Friendly, personal", "formal and academic": "Precise, scholarly",
        "playful and light": "Humorous, whimsical", "serious and reflective": "Thoughtful, weighty",
        "inspirational and motivating": "Encouraging, energetic", "neutral and instructional": "Plain, factual, how-to"}),
    "voice": _q("Which narrative voice does the text use?", {
        "first person singular": "I / my", "first person plural": "we / our", "second person": "you / your",
        "third person": "he / she / they / it", "impersonal": "No personal pronouns, e.g. documentation"}),
    "audience": _q("Who is the natural reader?", {
        "general readers": "Anyone", "children": "Young children", "young adults": "Teens and young adults",
        "students and learners": "People learning a subject", "professionals": "Practitioners at work",
        "specialists": "Experts in the field"}),
    "pace": _q("How long should each chapter be?", {
        "short": "About 1,500 words, e.g. children's books, listicles, short pieces",
        "medium": "About 3,000 words, the usual length",
        "long": "About 5,000 words, dense or technical material"}),
    "preface": {"type": "noul", "instructions": "Would a professional publisher add a preface to this book?"},
    "glossary": {"type": "noul", "instructions": "Does the text use specialised terms a reader would need defined?"},
}
PACE = {"short": 1500, "medium": 3000, "long": 5000}

TITLING = """You are a book editor. From the manuscript excerpts give: the book's title (reuse the author's own title \
if the text has one, otherwise craft a fitting one), an optional subtitle, a one-sentence subject, and the ISO 639-1 \
language code of the manuscript. Answer with JSON only."""

POLISH = """You are a senior book editor at a professional publishing house. Rewrite the SOURCE passage as \
publication-ready book prose.
- Preserve every idea, fact, example, name, number and the author's intent. Add nothing new; omit nothing.
- Keep the author's subject, tone ({tone}) and voice ({voice}). Genre: {genre}. Audience: {audience}.
- Write in the same language as the source (ISO code: {language}).
- Turn rough notes, fragments and scribbles into flowing, well-structured paragraphs. Keep real lists, code and tables as such.
- Fix grammar, flow and transitions; remove repetition and filler, never content. Keep the source's own headings as '## ' subheadings. {rule}
- Markdown only. Use '## ' for subheadings where the structure calls for them; never '# '. No chapter titles, no commentary.
- Continue seamlessly from the previous text; never repeat or summarize it."""

NUMRULE = {"keep": "Keep the source's own section numbers exactly as written; never add or change numbers.",
           "auto": "Never write section numbers in headings; numbering is added later.",
           "none": "Never write numbers in headings."}
NUMMODE = {"keep": "keep", "auto": "bare", "none": "strip"}

TITLER = ("You are a book editor. Give this chapter a short, accurate title (max 8 words, no 'Chapter N' prefix, "
          "different from titles already used) and a one-sentence summary. Answer with JSON only.")
GLOSSER = ("Extract up to 25 key terms that appear in the text and that a reader may need explained. Define each in one "
           "sentence, using only what the text says or the standard meaning. Write in the text's language. JSON only.")
PREFACER = ("You are the author's editor drafting a preface (200-300 words) in the book's own tone and voice: what the "
            "book is about, who it is for, how to read it. Use only the information given; never invent biography, "
            "credentials, anecdotes or claims. Markdown prose, no heading. Write in the book's language.")

REWRITE = """You are a senior book editor. Apply the EDIT INSTRUCTION to the SOURCE passage and return the revised \
passage in Markdown.
- Change only what the instruction asks for; keep every other idea, fact, name and the structure intact.
- Keep the tone ({tone}), voice ({voice}) and language ({language}) unless the instruction says otherwise. {rule}
- Use '## ' for subheadings; never '# '. No chapter title, no commentary, no code fence around your answer.
{notes}"""

NEWCHAPTER = """You are the author's editor writing the body of a NEW chapter that the author asked for.
- Cover what the instruction says. Do not invent facts, statistics, quotes or biography; stay general where unsure.
- Match the book's tone ({tone}), voice ({voice}), audience ({audience}) and language ({language}).
- Markdown, '## ' for subheadings, never '# '. No chapter title, no commentary. {rule}
{notes}"""

SUMMARISER = "Summarise this chapter in ONE sentence. Plain text only, same language as the chapter."

CHAT_SYS = """You are the Book Forge assistant: an editor and Q&A partner for ONE book that the user just generated. \
You only know the book through the OUTLINE and EXCERPTS you are given.
Reply with ONE JSON object: {"answer": str, "ops": [..], "remember": [..]}.
Rules:
- Questions / clarifications: answer in "answer" using only the outline and excerpts. If they do not contain the \
answer, say so and point to the chapter to check; never invent content. Leave "ops" empty.
- Change requests (edit, rewrite, rename, add, delete, move, fix a typo, change tone...): put each change in "ops" and \
say briefly in "answer" what you are doing. Emit only changes the user clearly asked for.
- Op fields: op, target, find, value, definition, instruction, position (unused fields stay empty / 0).
  set_title | set_subtitle | set_author : value
  rename_chapter : target (chapter number), value (new title)
  find_replace : target (chapter number, "preface" or "all"), find (text copied EXACTLY from the excerpts), value (replacement)
  rewrite : target (chapter number, "preface" or "all"), instruction (what to change, e.g. "make it more formal", "shorten by a third", "fix grammar")
  add_chapter : value (title), instruction (what it should cover), position (1-based place, 0 = at the end)
  delete_chapter : target
  move_chapter : target, position (new 1-based place)
  add_term : value (term), definition   |  remove_term : value
  regenerate_preface | regenerate_glossary
- If you cannot see the exact text to replace, use "rewrite" with an instruction instead of guessing a "find".
- Chapter numbers are 1-based and refer to the outline as it is NOW. List delete_chapter / move_chapter ops last.
- "remember": short durable preferences or decisions the user states for this book (e.g. "use British spelling"). Usually empty.
- Be concise and friendly. Answer in the user's language."""


class Op(BaseModel):
    op: str
    target: str = ""
    find: str = ""
    value: str = ""
    definition: str = ""
    instruction: str = ""
    position: int = 0


class Reply(BaseModel):
    answer: str = ""
    ops: list[Op] = []
    remember: list[str] = []


OPS = {"set_title", "set_subtitle", "set_author", "rename_chapter", "find_replace", "rewrite", "add_chapter",
       "delete_chapter", "move_chapter", "add_term", "remove_term", "regenerate_preface", "regenerate_glossary"}
STRUCTURAL = {"add_chapter", "delete_chapter", "move_chapter"}


# ───────────────────────── Ollama helpers ─────────────────────────
def ask(model, prompt, schema=None, system="", temp=0.3, predict=None):
    """One chat call; with `schema` the reply is validated into that Pydantic model (3 tries).
    A reply cut off by the token limit is never accepted: it is retried with a larger limit."""
    js = schema.model_json_schema() if schema else None
    if schema:  # schema in the prompt too: some models (e.g. gpt-oss) don't reliably honour `format=`
        system = (system + "\n\n" if system else "") + ("Reply with ONLY one JSON object that validates against this "
                                                        "JSON schema. No prose, no code fences:\n" + json.dumps(js))
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    err, out, pred = "", "", predict
    for attempt in range(3):
        m = msgs + ([{"role": "user", "content": f"Your previous reply was invalid ({err}). "
                                                 "Reply again with only the corrected JSON object."}] if err else [])
        # attempt 1 drops the grammar constraint, in case it is what produced an empty reply
        fmt = js if schema and attempt != 1 else None
        resp = chat_once(model, m, temp, fmt, pred)
        out = (resp.message.content or "").strip()
        cut = getattr(resp, "done_reason", None) == "length"
        if cut:  # ran out of tokens / context: never accept half an answer
            pred = (pred or 4096) * 2
        if not schema:
            if out and not cut:
                return out
            err = "reply was cut off (context or token limit)" if cut else \
                f"empty reply, done_reason={getattr(resp, 'done_reason', None)}"
            continue
        obj, err = parse_model(schema, out)
        if obj and not cut:
            return obj
        if cut:
            err = "reply was cut off (context or token limit)"
    raise RuntimeError(f"{model} returned unusable output ({err}). Last reply: {out[:200]!r}")


def chat_once(model, messages, temp, fmt, predict=None):
    """gpt-oss gets low reasoning effort (so hidden thinking can't eat the context); other models get none (faster)."""
    opts = {**ctx_for(model), "temperature": temp}
    if predict:
        opts["num_predict"] = predict
    kw = dict(model=model, messages=messages, options=opts, keep_alive="10m", format=fmt)
    try:
        return client.chat(**kw, think="low" if model.startswith("gpt-oss") else False)
    except ollama.ResponseError as ex:
        if "think" in str(ex).lower():  # older Ollama or a model that rejects the flag
            return client.chat(**kw)
        raise


def parse_model(schema, text: str):
    """Leniently pull a JSON object out of `text` and validate it with the Pydantic schema -> (model|None, error)."""
    text, err = re.sub(r"```(?:json)?", "", text), "no JSON object found"
    for m in re.finditer(r"\{", text):
        try:
            return schema.model_validate(json.JSONDecoder().raw_decode(text[m.start():])[0]), ""
        except ValueError as ex:  # JSON syntax or Pydantic validation error
            err = str(ex).splitlines()[0][:200]
    return None, err


def decide(model, state, questions) -> dict:
    """Ask the decision model (nimble) structured questions via Ollama's /v1/systemone endpoint."""
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").replace("0.0.0.0", "127.0.0.1")
    host = host if "://" in host else "http://" + host
    host = host if re.search(r":\d+$", host) else host + ":11434"
    try:
        r = httpx.post(f"{host.rstrip('/')}/v1/systemone", timeout=600,
                       json={"model": model, "state": state, "questions": questions, "keep_alive": "10m"})
    except httpx.HTTPError as ex:
        raise RuntimeError(f"Can't reach Ollama's decision endpoint: {ex}")
    if r.status_code != 200:
        raise RuntimeError(f"Decision call to '{model}' failed ({r.status_code}): {r.text[:200]}. "
                           "Use a decision model such as nimble and Ollama 0.35 or newer.")
    return r.json()["answers"]


def embed(model, texts, batch=16):
    v = np.vstack([np.array(client.embed(model=model, input=texts[i:i + batch], options=EMB_CTX, keep_alive="10m").embeddings, dtype=float)
                   for i in range(0, len(texts), batch)])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@st.cache_data(ttl=30)
def local_models() -> list[str]:
    return sorted(m.model for m in ollama.list().models)


def pick(label, names, prefs, key=None):
    i = next((names.index(n) for p in prefs for n in (p, p + ":latest") if n in names), 0)
    return st.selectbox(label, names, index=i, key=key)


# ───────────────────────── Input: extract, clean, chunk ─────────────────────────
def docx_text(data: bytes) -> str:
    from docx import Document
    from docx.table import Table as DTable
    from docx.text.paragraph import Paragraph as DPara
    d, out = Document(io.BytesIO(data)), []
    for el in d.element.body.iterchildren():
        if el.tag.endswith("}p"):
            p = DPara(el, d)
            s = p.style.name if p.style is not None else ""
            m = re.match(r"Heading (\d)", s)
            out.append(("#" * int(m[1]) + " " if m else "# " if s == "Title" else "") + p.text)
        elif el.tag.endswith("}tbl"):
            rows = [[c.text.strip() for c in r.cells] for r in DTable(el, d).rows]
            if rows:
                lines = ["| " + " | ".join(r) + " |" for r in rows]
                lines.insert(1, "|" + "---|" * len(rows[0]))
                out.append("\n".join(lines))
    return "\n\n".join(out)


# A numbered section heading on its own line: "1 Introduction", "5.6 Results", "## 2.1 Setup"
HEAD = re.compile(r"^[ \t]*(?:#{1,6}[ \t]+)?\d{1,2}(?:\.\d{1,2}){0,3}\.?[ \t]+[A-Z][^\n.;,:]{2,80}$", re.M)


def numbered_source(text: str) -> bool:
    return len(HEAD.findall(text)) >= 4


def numbers_consistent(text: str) -> bool:
    """True only if the source's own section numbers run in a clean ascending order (no repeats, jumps back or noise)."""
    seq = []
    for h in HEAD.findall(text):
        n = re.match(r"[ \t]*(?:#{1,6}[ \t]+)?(\d{1,2}(?:\.\d{1,2}){0,3})", h)
        if n:
            seq.append(tuple(int(x) for x in n[1].split(".")))
    return len(seq) >= 4 and sum(a < b for a, b in zip(seq, seq[1:])) / (len(seq) - 1) >= 0.95


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    t = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    t = re.sub(r"-\n(?=[a-z])", "", t)                         # de-hyphenate
    t = HEAD.sub(lambda m: "\n" + m.group(0).strip() + "\n", t)  # numbered headings stay on their own line
    t = re.sub(r"(?<![.!?:\"”)\n])\n(?!\n)", " ", t)           # re-join wrapped lines
    return re.sub(r"(?<!\n)\n(?!\n)", "\n\n", t)               # remaining breaks are paragraphs


# Typographic look-alikes that PDFs/Word emit and that standard PDF fonts draw as black boxes
NORM = {0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2043: "-", 0x2212: "-", 0xAD: "", 0x200B: "", 0xFEFF: "",
        0xA0: " ", 0x2009: " ", 0x202F: " ", 0x2002: " ", 0x2003: " ",
        0xFB00: "ff", 0xFB01: "fi", 0xFB02: "fl", 0xFB03: "ffi", 0xFB04: "ffl"}


def normalize(t: str) -> str:
    return t.translate(NORM)


def extract_text(name: str, data: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return normalize(pdf_text(data))
    if ext == ".docx":
        return normalize(docx_text(data))
    if ext == ".doc":
        raise ValueError(f"{name}: legacy .doc isn't supported. Save it as .docx and upload again.")
    return normalize(data.decode("utf-8", errors="replace"))


def split_long(p: str, n: int) -> list[str]:
    pieces, cur = [], []
    for s in re.findall(r"[^.!?]+[.!?]*\s*", p):
        w = s.split()
        for i in range(0, len(w), n):
            seg = " ".join(w[i:i + n])
            if cur and sum(len(x.split()) for x in cur) + len(seg.split()) > n:
                pieces.append(" ".join(cur))
                cur = []
            cur.append(seg)
    return pieces + [" ".join(cur)] if cur else pieces


def chunk(text: str, n: int = CHUNK_WORDS) -> list[str]:
    units = []
    for p in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if p:
            units += split_long(p, n) if len(p.split()) > n else [p]
    chunks, cur, size = [], [], 0
    for u in units:
        w = len(u.split())
        if cur and size + w > n:
            chunks.append("\n\n".join(cur))
            cur, size = [], 0
        cur.append(u)
        size += w
    return chunks + ["\n\n".join(cur)] if cur else chunks


def segment(vecs: np.ndarray, k: int) -> list[list[int]]:
    """Split ordered chunks into k chapters at the weakest topical seams (windowed cosine, TextTiling style)."""
    n = len(vecs)
    k = max(1, min(k, n))
    if k == 1:
        return [list(range(n))]
    gaps = [float(vecs[max(0, i - 2):i].mean(0) @ vecs[i:i + 2].mean(0)) for i in range(1, n)]
    cuts = sorted(int(i) + 1 for i in np.argsort(gaps)[:k - 1])
    b = [0, *cuts, n]
    return [list(range(a, c)) for a, c in zip(b, b[1:])]


# ───────────────────────── Pipeline ─────────────────────────
def clean(t: str, mode: str = "strip") -> str:
    """mode: keep = leave heading numbers alone, strip = drop them, bare = also drop '1 Introduction'-style numbers."""
    t = t.strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\s*\n|\n```\s*$", "", t)
    parts = re.split(r"(```.*?```)", t, flags=re.S)  # demote stray '# ' headings, but not code comments

    def fix(p):
        p = tidy_headings(re.sub(r"^#\s", "## ", normalize(p), flags=re.M))
        return p if mode == "keep" else strip_numbers(p, mode == "bare")
    return "".join(p if p.startswith("```") else fix(p) for p in parts)


NUM = r"(?:(?:chapter|section|part)\s+\d+\s*[.:)\-]?\s*|\d+(?:\.\d+)+\.?\s+|\d+[.):]\s+)"


def strip_numbers(t: str, bare: bool = False) -> str:
    """Drop numbering the source or the model put in headings; `bare` also drops '1 Introduction' (digit, no dot)."""
    pat = "(?:" + NUM + (r"|\d{1,2}\s+(?=[A-Z])|[A-Z]\.?\s?\d+(?:\.\d+)*\.?\s+(?=[A-Z])" if bare else "") + ")"
    return re.sub(r"^(#{1,6}\s+)(?:\*\*)?" + pat, r"\1", t, flags=re.M | re.I)


def faithful(decider, src, out) -> float:
    """Decision model's yes/no verdict (0-1) on whether the rewrite keeps the source's meaning."""
    if (len(src.split()) + len(out.split())) * 1.6 > 6000:  # nimble's prompt must fit in 8,192 tokens
        return 1.0  # too long to check; the embedding check still applies
    q = {"ok": {"type": "noul", "instructions": "Does the rewrite keep every idea, fact and name of the original "
                                                "and add no new claims?"}}
    return float(decide(decider, {"original": src, "rewrite": out}, q)["ok"]["noul"])


def polish(src, prev, bp, writer, emb, decider=None, depth=0):
    """Polish one chunk -> (text, similarity or None if kept as-is, failure note).
    Meaning is verified with embeddings; the decision model is only consulted when similarity is borderline."""
    system, mode = POLISH.format(**bp.model_dump(), rule=NUMRULE[bp.numbering]), NUMMODE[bp.numbering]
    best, best_s, best_sim = "", -9.0, 0.0
    for attempt in range(2):
        extra = "" if not attempt else ("\nYour previous attempt drifted from the source (altered meaning or dropped "
                                        "content). Stay much closer to the source and keep every point.")
        try:
            out = clean(ask(writer, f"BOOK SUBJECT: {bp.subject}\n\nPREVIOUS TEXT (continuity only, do not repeat):\n"
                                    f"{prev or '(start of book)'}\n\nSOURCE:\n{src}", system=system + extra), mode)
        except RuntimeError as ex:
            if best:
                break
            words = len(src.split())
            if depth == 0 and words > 250:  # small models cope better with half the text: retry in two pieces
                parts = [polish(h, prev if k == 0 else "", bp, writer, emb, decider, 1)
                         for k, h in enumerate(chunk(src, words // 2 + 1))]
                got = [x[1] for x in parts if x[1] is not None]
                return ("\n\n".join(x[0] for x in parts), float(np.mean(got)) if got else None,
                        next((x[2] for x in parts if x[2]), ""))
            return clean(src, mode), None, str(ex)[:160]  # keep the author's text rather than abort the whole book
        v = embed(emb, [src, out])
        sim, ratio = float(v[0] @ v[1]), len(out.split()) / max(1, len(src.split()))
        f = faithful(decider, src, out) if decider and sim < 0.9 else 1.0
        if sim + f > best_s:
            best, best_sim, best_s = out, sim, sim + f
        if sim >= MIN_SIM and 0.6 <= ratio <= 2.5 and f >= 0.5:
            break
    return best, best_sim, ""


def make_preface(writer, bp, chapters) -> str:
    toc = "\n".join(f"{i}. {c.title}: {c.summary}" for i, c in enumerate(chapters, 1))
    return clean(ask(writer, f"{bp.model_dump_json()}\n\nChapters:\n{toc}", system=PREFACER, temp=0.5))


def make_glossary(writer, chapters) -> list[Term]:
    sample = "\n\n".join(c.body[:1500] for c in chapters)[:14000]
    try:
        return ask(writer, sample, Glossary, GLOSSER).terms
    except RuntimeError:
        return []


def make_book(text, decider, writer, emb, title, author, bar, verify=True, workers=1) -> Book:
    t0, chunks, words = time.time(), chunk(text), len(text.split())
    bar.progress(0.02, "Reading the manuscript…")
    step = max(1, len(chunks) // 12)
    digest = "\n\n---\n\n".join(c[:700] for c in chunks[::step][:12])
    a = decide(decider, {"word_count": words, "manuscript_excerpts": digest}, QUESTIONS)  # nimble: choices only
    t = ask(writer, f"Manuscript: {words} words. Sample excerpts:\n\n{digest}", Titling, TITLING)  # writer: free text
    bp = Blueprint(title=title or t.title, subtitle=t.subtitle, subject=t.subject, language=t.language,
                   genre=a["genre"]["choice"], tone=a["tone"]["choice"], voice=a["voice"]["choice"],
                   audience=a["audience"]["choice"], chapter_count=max(1, round(words / PACE[a["pace"]["choice"]])),
                   include_preface=a["preface"]["noul"] >= .5, include_glossary=a["glossary"]["noul"] >= .5)
    # keep the author's own section numbers if they have them (cross-references like §5.6 depend on them)
    # (messy numbers, e.g. from a PDF paper, would give "2 Setup, 2 Preliminaries, 3.2 ..." so those get renumbered)
    bp.numbering = ("keep" if numbered_source(text) and numbers_consistent(text)
                    else "auto" if numbered_source(text) or bp.genre in NUMBERED else "none")

    bar.progress(0.08, "Mapping the structure…")
    groups = segment(embed(emb, chunks), bp.chapter_count)

    # passages are independent (continuity comes from the preceding source text), so they can run in parallel
    prevs, res, n = [""] + [c[-700:] for c in chunks[:-1]], {}, 0
    ex = ThreadPoolExecutor(max_workers=max(1, workers))
    futs = {ex.submit(polish, chunks[i], prevs[i], bp, writer, emb, decider if verify else None): i
            for i in range(len(chunks))}
    try:
        for f in as_completed(futs):
            res[futs[f]], n = f.result(), n + 1
            el = time.time() - t0
            left = el / n * (len(chunks) - n)
            bar.progress(0.1 + 0.75 * n / len(chunks),
                         f"Polishing part {n}/{len(chunks)} · {int(el // 60)}m{int(el % 60):02d}s elapsed, "
                         f"about {int(left // 60)}m left")
    except BaseException:
        for f in futs:
            f.cancel()
        raise
    finally:
        ex.shutdown(wait=False)
    sims = [r[1] for r in res.values() if r[1] is not None]
    skipped, why = sum(r[1] is None for r in res.values()), next((r[2] for r in res.values() if r[2]), "")

    chapters = []
    for g in groups:
        body = [res[i][0] for i in g]
        text_ = "\n\n".join(body)
        used = ", ".join(c.title for c in chapters) or "none"
        try:
            meta = ask(writer, f"Book: {bp.title} ({bp.genre}). Titles already used: {used}.\n\nChapter text:\n{text_[:2500]}",
                       ChapterMeta, TITLER)
        except RuntimeError:
            meta = ChapterMeta(title=f"Part {len(chapters) + 1}", summary="")
        chapters.append(Chapter(title=re.sub(NUM, "", meta.title.strip(), flags=re.I) or meta.title,
                                summary=meta.summary, body=text_))

    preface, glossary = "", []
    if bp.include_preface:
        bar.progress(0.88, "Writing the preface…")
        preface = make_preface(writer, bp, chapters)
    if bp.include_glossary:
        bar.progress(0.93, "Compiling the glossary…")
        glossary = make_glossary(writer, chapters)
    bar.progress(1.0, "Done")
    return Book(blueprint=bp, author=author, preface=preface, chapters=chapters, glossary=glossary,
                fidelity=float(np.mean(sims)) if sims else 0.0, unpolished=skipped, why=why)


# ───────────────────────── Export: PDF ─────────────────────────
def _style(name, font="Times-Roman", **kw):
    return ParagraphStyle(name, fontName=font, **kw)


S = {
    "body": _style("body", fontSize=11, leading=16, alignment=TA_JUSTIFY, spaceAfter=8),
    "h1": _style("H1", "Times-Bold", fontSize=24, leading=30, spaceAfter=24, pageBreakBefore=1),
    "toch": _style("TOCH", "Times-Bold", fontSize=24, leading=30, spaceAfter=24, pageBreakBefore=1),
    "h2": _style("h2", "Times-Bold", fontSize=15, leading=20, spaceBefore=12, spaceAfter=8, keepWithNext=1),
    "h3": _style("h3", "Times-BoldItalic", fontSize=12.5, leading=17, spaceBefore=8, spaceAfter=6, keepWithNext=1),
    "li": _style("li", fontSize=11, leading=16, leftIndent=22, bulletIndent=8, spaceAfter=4),
    "quote": _style("quote", "Times-Italic", fontSize=11, leading=16, leftIndent=24, rightIndent=12, spaceAfter=8),
    "code": _style("code", "Courier", fontSize=8.5, leading=11, leftIndent=8, spaceAfter=8, backColor=colors.whitesmoke),
    "cell": _style("cell", fontSize=9.5, leading=12),
    "title": _style("title", "Times-Bold", fontSize=32, leading=38, alignment=TA_CENTER, spaceAfter=14),
    "sub": _style("sub", "Times-Italic", fontSize=15, leading=20, alignment=TA_CENTER, spaceAfter=40),
    "author": _style("author", fontSize=14, alignment=TA_CENTER),
}


def pdf_safe(t: str) -> str:
    """Standard PDF fonts only draw Windows-1252; map anything else to a look-alike so no black boxes appear."""
    out = []
    for c in normalize(t):
        try:
            c.encode("cp1252")
            out.append(c)
        except UnicodeEncodeError:
            out.append({"\u2192": "->", "\u2190": "<-", "\u2713": "v", "\u2264": "<=", "\u2265": ">="}.get(c)
                       or unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode() or "?")
    return "".join(out)


def inline(t: str) -> str:
    t = html.escape(pdf_safe(t), quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", t)
    return re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", t)


def safe_para(text: str, style, **kw):
    """A Paragraph that never aborts the export: if the inline markup can't be parsed (e.g. mis-nested **bold** and
    *italic*), fall back to the same text as plain, escaped characters."""
    try:
        return Paragraph(inline(text), style, **kw)
    except Exception:
        return Paragraph(html.escape(pdf_safe(text), quote=False), style, **kw)


def make_table(rows: list[list[str]], width: float):
    """Markdown table -> ReportLab Table with explicit column widths (never left to ReportLab's auto-sizing).
    Returns None when the table has too many columns to be readable on a 6x9 page."""
    n = max(map(len, rows))
    if n > 7:
        return None
    rows = [r + [""] * (n - len(r)) for r in rows]
    weights = [min(40, max(4, max(len(r[j]) for r in rows))) for j in range(n)]
    floor = 0.45 * inch
    w = [max(floor, width * x / sum(weights)) for x in weights]
    w = [x * width / sum(w) for x in w]  # scale so the table is exactly as wide as the text frame
    data = [[safe_para(c.strip() or " ", S["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=w, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return t


def md_flow(md: str, width: float = 4.3 * inch, plain_tables: bool = False) -> list:
    """Markdown -> ReportLab flowables. plain_tables=True is the safe mode: tables become wrapped text lines."""
    out, para, rows, code = [], [], [], None

    def flush():
        if para:
            out.append(safe_para(" ".join(para), S["body"]))
            para.clear()
        if rows:
            t = None if plain_tables else make_table(rows, width)
            if t is not None:
                out.extend([t, Spacer(1, 8)])
            else:  # too wide, or safe mode: one wrapped line per row
                out.extend(safe_para("  |  ".join(c for c in r if c.strip()), S["body"]) for r in rows)
            rows.clear()

    for line in md.splitlines():
        s = line.strip()
        if s.startswith("```"):
            flush()
            if code is None:
                code = []
            else:
                out.append(Preformatted(pdf_safe("\n".join(code)), S["code"]))
                code = None
            continue
        if code is not None:
            code.append(line)
        elif s.startswith("|"):
            if para:
                out.append(safe_para(" ".join(para), S["body"]))
                para.clear()
            if not re.fullmatch(r"[\s|:\-]+", s):
                rows.append([c.strip() for c in s.strip("|").split("|")])
        else:
            flush()
            if not s:
                continue
            if m := re.match(r"(#{1,4})\s+(.*)", s):
                out.append(safe_para(m[2], S["h2"] if len(m[1]) <= 2 else S["h3"]))
            elif m := re.match(r"([-*+]|\d+[.)])\s+(.*)", s):
                out.append(safe_para(m[2], S["li"], bulletText="•" if m[1][0] in "-*+" else m[1]))
            elif s.startswith(">"):
                out.append(safe_para(s.lstrip("> "), S["quote"]))
            else:
                para.append(s)
    flush()
    return out + ([Preformatted(pdf_safe("\n".join(code)), S["code"])] if code else [])


class _Doc(BaseDocTemplate):
    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name == "H1":
            txt, key = f.getPlainText(), f"s{self.page}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(txt, key, 0)
            self.notify("TOCEntry", (0, txt, self.page))


def _build_pdf(book: Book, plain_tables: bool) -> bytes:
    b, buf = book.blueprint, io.BytesIO()
    doc = _Doc(buf, pagesize=(6 * inch, 9 * inch), leftMargin=.85 * inch, rightMargin=.85 * inch,
               topMargin=.9 * inch, bottomMargin=.9 * inch, title=b.title, author=book.author)

    def footer(c, d):
        if d.page > 1:
            c.setFont("Times-Roman", 9)
            c.drawCentredString(3 * inch, .5 * inch, str(d.page))

    doc.addPageTemplates([PageTemplate("p", [Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")],
                                       onPage=footer)])
    toc = TableOfContents()
    toc.levelStyles = [_style("toc", fontSize=11, leading=20)]
    story = [Spacer(1, 2 * inch), safe_para(b.title, S["title"])]
    if b.subtitle:
        story.append(safe_para(b.subtitle, S["sub"]))
    if book.author:
        story.append(safe_para(book.author, S["author"]))
    story += [Paragraph("Contents", S["toch"]), toc]
    for title, md in book.sections():
        story += [safe_para(title, S["h1"]), *md_flow(md, doc.width, plain_tables)]
    doc.multiBuild(story)
    return buf.getvalue()


def build_pdf(book: Book) -> bytes:
    """Normal layout first; if ReportLab chokes on something (an oversized table row, odd markup...), build again
    with tables as plain text lines so one awkward passage can never block the whole export."""
    try:
        return _build_pdf(book, plain_tables=False)
    except Exception:
        return _build_pdf(book, plain_tables=True)


# ───────────────────────── Export: EPUB ─────────────────────────
CSS = """body{font-family:Georgia,serif;line-height:1.6;margin:5%}h1{margin:2em 0 1em;text-align:center}
h2{margin-top:1.6em}p{text-align:justify;margin:.6em 0}blockquote{font-style:italic;margin:1em 2em}
pre,code{font-family:monospace;font-size:.85em;background:#f4f4f4}table{border-collapse:collapse}
td,th{border:1px solid #999;padding:4px 8px}.t{text-align:center;margin-top:30%}"""


def build_epub(book: Book) -> bytes:
    import markdown
    from ebooklib import epub
    b = book.blueprint
    e = epub.EpubBook()
    e.set_identifier(str(uuid.uuid4()))
    e.set_title(b.title)
    e.set_language(b.language or "en")
    if book.author:
        e.add_author(book.author)
    css = epub.EpubItem(uid="css", file_name="style.css", media_type="text/css", content=CSS)
    e.add_item(css)
    cover = f"<div class='t'><h1>{html.escape(b.title)}</h1>"
    cover += f"<h3><i>{html.escape(b.subtitle)}</i></h3>" if b.subtitle else ""
    cover += f"<p style='text-align:center'>{html.escape(book.author)}</p></div>"
    pages = [epub.EpubHtml(title="Title Page", file_name="title.xhtml", content=cover)]
    for i, (title, md) in enumerate(book.sections()):
        body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
        pages.append(epub.EpubHtml(title=title, file_name=f"s{i}.xhtml", content=f"<h1>{html.escape(title)}</h1>{body}"))
    for p in pages:
        p.add_item(css)
        e.add_item(p)
    e.toc = pages[1:]
    e.spine = ["nav", *pages]
    e.add_item(epub.EpubNcx())
    e.add_item(epub.EpubNav())
    buf = io.BytesIO()
    epub.write_epub(buf, e)
    return buf.getvalue()


def full_markdown(book: Book) -> str:
    b = book.blueprint
    head = f"# {b.title}\n\n" + (f"*{b.subtitle}*\n\n" if b.subtitle else "") + (f"**{book.author}**\n\n" if book.author else "")
    return head + "\n\n".join(f"# {t}\n\n{md}" for t, md in book.sections()) + "\n"


# ───────────────────────── Session state, undo, project files ─────────────────────────
def new_memory() -> dict:
    return {"notes": [], "summary": "", "upto": 0}


def init_state():
    for k, v in dict(rev=0, mrev=0, undo=[], chat=[], pending=[]).items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("memory", new_memory())


def set_book(book: Book):
    """Load a (new or opened) book and reset everything that belongs to the previous one."""
    st.session_state.update(book=book, original=book.model_dump_json(), undo=[], chat=[], pending=[],
                            memory=new_memory())
    st.session_state.rev += 1
    st.session_state.mrev += 1
    for k in [k for k in st.session_state if k.startswith(("exp_", "gl_src_"))]:
        del st.session_state[k]


def snap():
    u = st.session_state.setdefault("undo", [])
    u.append(st.session_state.book.model_dump_json())
    del u[:-30]


def bump_rev():
    st.session_state.rev += 1


def drafted_book(book: Book) -> Book:
    """A copy of the book with everything currently typed / selected in the Edit tab applied. Nothing is saved yet."""
    rev, d = st.session_state.rev, book.model_copy(deep=True)
    bp = d.blueprint

    def g(name, cur):
        return st.session_state.get(f"{name}_{rev}", cur)
    bp.title = g("e_title", bp.title).strip() or bp.title
    bp.subtitle = g("e_sub", bp.subtitle)
    d.author = g("e_author", d.author)
    bp.numbering = g("e_num", bp.numbering)
    d.preface = g("e_pref", d.preface)
    for c in d.chapters:
        c.title = g(f"e_ct_{c.id}", c.title).strip() or c.title
        c.body = g(f"e_cb_{c.id}", c.body)
    d.glossary = st.session_state.get(f"gl_cur_{rev}", d.glossary)
    return d


def has_drafts(book: Book) -> bool:
    return drafted_book(book).model_dump_json() != book.model_dump_json()


def sync_widgets():
    """Commit the Edit tab's drafts into the book (one undo step). No-op when nothing changed."""
    book = st.session_state.get("book")
    if book is None:
        return
    new = drafted_book(book)
    if new.model_dump_json() != book.model_dump_json():
        snap()
        st.session_state.book = new


def cb_apply():
    sync_widgets()
    bump_rev()  # rebuild the editor from the saved book
    st.session_state.flash = "Changes applied. The preview and exports now use them."


def cb_discard():
    bump_rev()
    st.session_state.flash = "Unapplied edits discarded."


def cb_undo():
    if st.session_state.undo:
        st.session_state.book = Book.model_validate_json(st.session_state.undo.pop())
        bump_rev()


def cb_revert():
    snap()
    st.session_state.book = Book.model_validate_json(st.session_state.original)
    bump_rev()


def cb_move(i, j):
    sync_widgets()
    snap()
    ch = st.session_state.book.chapters
    ch.insert(j, ch.pop(i))
    bump_rev()


def cb_delete(i):
    sync_widgets()
    ch = st.session_state.book.chapters
    if len(ch) > 1:
        snap()
        ch.pop(i)
        bump_rev()


def cb_add_chapter():
    sync_widgets()
    snap()
    st.session_state.book.chapters.append(Chapter(title="New chapter", body="Write here…"))
    bump_rev()


def cached_export(book, fmt):
    c = st.session_state.get("exp_" + fmt)
    return c[1] if c and c[0] == hashlib.sha1(book.model_dump_json().encode()).hexdigest() else None


def make_export(book, fmt):
    data = build_pdf(book) if fmt == "pdf" else build_epub(book)
    st.session_state["exp_" + fmt] = (hashlib.sha1(book.model_dump_json().encode()).hexdigest(), data)
    return data


def project_json(book) -> str:
    return json.dumps({"app": "bookforge", "version": 2, "book": json.loads(book.model_dump_json()),
                       "chat": st.session_state.chat, "memory": st.session_state.memory}, ensure_ascii=False)


# ───────────────────────── Assistant: context, retrieval, actions ─────────────────────────
def words(t: str) -> int:
    return len(t.split())


def notes_text() -> str:
    n = st.session_state.memory["notes"]
    return "AUTHOR'S STANDING PREFERENCES (follow them):\n" + "\n".join(f"- {x}" for x in n) if n else ""


def book_outline(book: Book, cap: int = 4000) -> str:
    b = book.blueprint
    lines = [f"Title: {b.title}" + (f" - {b.subtitle}" if b.subtitle else ""),
             f"Author: {book.author or '(not set)'}",
             f"Genre: {b.genre}; tone: {b.tone}; voice: {b.voice}; audience: {b.audience}; language: {b.language}",
             f"Subject: {b.subject}",
             f"Preface: {'yes, ~%d words' % words(book.preface) if book.preface.strip() else 'none'}",
             "Chapters:"]
    for i, c in enumerate(book.chapters, 1):
        heads = re.findall(r"^##\s+(.+)$", c.body, re.M)[:6]
        lines.append(f"{i}. {c.title} (~{words(c.body)} words)" + (f" - {c.summary}" if c.summary else "")
                     + (f" [sections: {'; '.join(heads)}]" if heads else ""))
    if book.glossary:
        lines.append("Glossary terms: " + ", ".join(t.term for t in book.glossary[:40]))
    return "\n".join(lines)[:cap]


def passages(book: Book) -> list[tuple[str, str]]:
    out = [("Preface", p) for p in chunk(book.preface, 160)]
    for i, c in enumerate(book.chapters, 1):
        out += [(f"Chapter {i}: {c.title}", p) for p in chunk(c.body, 160)]
    out += [("Glossary", f"{t.term}: {t.definition}") for t in book.glossary]
    return out


def retrieve(book: Book, q: str, emb: str, k: int = 5) -> list[tuple[str, str]]:
    ps = passages(book)
    if not ps:
        return []
    try:  # semantic search; passage vectors are cached so only edited text is re-embedded
        cache = st.session_state.setdefault("vcache", {})
        keys = [f"{emb}|{hashlib.sha1(t.encode()).hexdigest()}" for _, t in ps]
        miss = [i for i, key in enumerate(keys) if key not in cache]
        if miss:
            for i, row in zip(miss, embed(emb, [ps[i][1] for i in miss])):
                cache[keys[i]] = row
        order = np.argsort(-(np.vstack([cache[x] for x in keys]) @ embed(emb, [q])[0]))[:k]
    except Exception:  # embedding model unavailable: keyword overlap
        qw = set(re.findall(r"\w{4,}", q.lower()))
        order = sorted(range(len(ps)), key=lambda i: -len(qw & set(re.findall(r"\w{4,}", ps[i][1].lower()))))[:k]
    return [ps[int(i)] for i in sorted(order)]


def build_excerpts(book: Book, msg: str, emb: str, cap: int = 8000, k: int = 5) -> str:
    parts = []
    if book.preface.strip() and re.search(r"preface|foreword|introduction to the book", msg, re.I):
        parts.append("[Preface (start)]\n" + " ".join(book.preface.split()[:350]))
    for m in dict.fromkeys(re.findall(r"chapter\s+(\d+)", msg, re.I)[:2]):
        idx = int(m) - 1
        if 0 <= idx < len(book.chapters):
            c, cs = book.chapters[idx], chunk(book.chapters[idx].body, 170)
            picks = [cs[round(j * (len(cs) - 1) / max(1, min(4, len(cs)) - 1))] for j in range(min(4, len(cs)))] if cs else []
            parts += [f"[Chapter {idx + 1}: {c.title} (sampled)]\n{p}" for p in dict.fromkeys(picks)]
    parts += [f"[{lab}]\n{t}" for lab, t in retrieve(book, msg, emb, k)]
    return "\n\n".join(parts)[:cap]


def transcript(msgs, each: int = 700) -> str:
    out = []
    for m in msgs:
        t = m["content"][:each] + (" [changes applied: " + "; ".join(m["actions"])[:300] + "]" if m.get("actions") else "")
        out.append(f"{'User' if m['role'] == 'user' else 'Assistant'}: {t}")
    return "\n".join(out) or "(none)"


def sub_text(text: str, find: str, new: str) -> tuple[str, int]:
    k = text.count(find)
    if k:
        return text.replace(find, new), k
    pat = re.compile(r"\s+".join(map(re.escape, find.split())), re.I)  # forgiving about whitespace / case
    return pat.subn(lambda m: new, text)


def rewrite_text(text: str, instruction: str, bp: Blueprint, model: str) -> str:
    system = REWRITE.format(**bp.model_dump(), rule=NUMRULE[bp.numbering], notes=notes_text())
    mode, out = NUMMODE[bp.numbering], []
    for part in chunk(text, 700):
        try:
            new = clean(ask(model, f"EDIT INSTRUCTION: {instruction}\n\nSOURCE:\n{part}", system=system, temp=0.3), mode)
        except RuntimeError:
            new = ""
        out.append(new or part)  # never lose text if the model fails
    return "\n\n".join(out)


def refresh_summary(ch: Chapter, model: str):
    try:
        ch.summary = ask(model, ch.body[:2500], system=SUMMARISER).strip()[:300]
    except RuntimeError:
        pass


def parse_target(t: str, n: int):
    t = (t or "").strip().lower()
    if t in ("all", "book", "everything", "*"):
        return "all"
    if t.startswith("pref"):
        return "preface"
    m = re.search(r"\d+", t)
    return int(m[0]) - 1 if m and 0 <= int(m[0]) - 1 < n else None


def apply_op(book: Book, o: Op, model: str) -> str:
    bp, n, kind = book.blueprint, len(book.chapters), o.op
    t = parse_target(o.target, n)
    if kind == "set_title":
        bp.title = o.value.strip() or bp.title
        return f"Title set to “{bp.title}”"
    if kind == "set_subtitle":
        bp.subtitle = o.value.strip()
        return "Subtitle updated"
    if kind == "set_author":
        book.author = o.value.strip()
        return f"Author set to “{book.author}”"
    if kind == "rename_chapter":
        if not isinstance(t, int) or not o.value.strip():
            raise ValueError("need a valid chapter number and a new title")
        book.chapters[t].title = o.value.strip()
        return f"Chapter {t + 1} renamed to “{o.value.strip()}”"
    if kind == "find_replace":
        if not o.find.strip():
            raise ValueError("nothing to find")
        total = 0

        def sub(s):
            nonlocal total
            s, k = sub_text(s, o.find, o.value)
            total += k
            return s
        if t in (None, "all"):
            bp.title, bp.subtitle, book.preface = sub(bp.title), sub(bp.subtitle), sub(book.preface)
            for c in book.chapters:
                c.title, c.body = sub(c.title), sub(c.body)
            for g in book.glossary:
                g.term, g.definition = sub(g.term), sub(g.definition)
        elif t == "preface":
            book.preface = sub(book.preface)
        else:
            book.chapters[t].body = sub(book.chapters[t].body)
        if not total:
            raise ValueError(f"couldn't find “{o.find[:60]}”")
        return f"Replaced “{o.find[:40]}” with “{o.value[:40]}” ({total}×)"
    if kind == "rewrite":
        ins = (o.instruction or o.value).strip()
        if not ins or t is None:
            raise ValueError("need a target and an instruction")
        if t == "preface":
            book.preface = rewrite_text(book.preface, ins, bp, model)
            return f"Preface rewritten: {ins[:60]}"
        targets = book.chapters if t == "all" else [book.chapters[t]]
        if t == "all" and book.preface.strip():
            book.preface = rewrite_text(book.preface, ins, bp, model)
        for c in targets:
            c.body = rewrite_text(c.body, ins, bp, model)
            refresh_summary(c, model)
        return (f"All chapters rewritten: {ins[:60]}" if t == "all" else f"Chapter {t + 1} rewritten: {ins[:60]}")
    if kind == "add_chapter":
        title = o.value.strip() or "New chapter"
        body = clean(ask(model, f"BOOK OUTLINE:\n{book_outline(book, 2500)}\n\nNEW CHAPTER TITLE: {title}\n"
                                f"INSTRUCTION: {o.instruction or 'Write the chapter.'}",
                         system=NEWCHAPTER.format(**bp.model_dump(), rule=NUMRULE[bp.numbering], notes=notes_text()),
                         temp=0.5), NUMMODE[bp.numbering])
        ch = Chapter(title=title, body=body)
        refresh_summary(ch, model)
        pos = o.position - 1 if 1 <= o.position <= n else n
        book.chapters.insert(pos, ch)
        return f"Added chapter “{title}” at position {pos + 1}"
    if kind == "delete_chapter":
        if not isinstance(t, int):
            raise ValueError("need a valid chapter number")
        if n <= 1:
            raise ValueError("a book needs at least one chapter")
        return f"Deleted chapter {t + 1}: “{book.chapters.pop(t).title}”"
    if kind == "move_chapter":
        if not isinstance(t, int) or not 1 <= o.position <= n:
            raise ValueError("need a valid chapter number and destination")
        book.chapters.insert(o.position - 1, book.chapters.pop(t))
        return f"Moved chapter {t + 1} to position {o.position}"
    if kind == "add_term":
        if not o.value.strip():
            raise ValueError("no term given")
        book.glossary = [g for g in book.glossary if g.term.lower() != o.value.strip().lower()]
        book.glossary.append(Term(term=o.value.strip(), definition=o.definition.strip()))
        return f"Glossary: added “{o.value.strip()}”"
    if kind == "remove_term":
        k = len(book.glossary)
        book.glossary = [g for g in book.glossary if g.term.lower() != o.value.strip().lower()]
        if len(book.glossary) == k:
            raise ValueError(f"no glossary term “{o.value}”")
        return f"Glossary: removed “{o.value.strip()}”"
    if kind == "regenerate_preface":
        book.preface = make_preface(model, bp, book.chapters)
        return "Preface regenerated"
    if kind == "regenerate_glossary":
        book.glossary = make_glossary(model, book.chapters) or book.glossary
        return "Glossary regenerated"
    raise ValueError(f"unknown action {kind}")


def apply_ops(book: Book, ops: list[Op], model: str) -> list[str]:
    sync_widgets()  # keep anything the user typed but hasn't applied yet
    book = st.session_state.book
    snap()  # one undo step for the whole batch
    results = []
    with use_ctx(SETTINGS["chat_ctx"]):  # rewrites and new chapters need room for long output
        for o in sorted(ops, key=lambda o: o.op in STRUCTURAL):  # content edits first, then add / delete / move
            try:
                results.append("✅ " + apply_op(book, o, model))
            except Exception as ex:
                results.append(f"⚠️ {o.op} not applied: {str(ex)[:150]}")
    bump_rev()
    return results


def describe_op(o: Op) -> str:
    detail = o.instruction or o.value or o.find
    return f"{o.op}" + (f" [{o.target}]" if o.target else "") + (f": {detail[:80]}" if detail else "")


def add_notes(notes: list[str]):
    mem = st.session_state.memory
    have = {x.lower() for x in mem["notes"]}
    new = [n.strip() for n in notes if n.strip() and n.strip().lower() not in have]
    if new:
        mem["notes"] = (mem["notes"] + new)[-25:]
        st.session_state.mrev += 1


def roll_memory(model: str):
    """Fold older conversation turns into a running summary so the assistant remembers without a huge prompt."""
    mem, chat = st.session_state.memory, st.session_state.chat
    if len(chat) - mem["upto"] <= 14:
        return
    cut = len(chat) - 6
    try:
        with use_ctx(SETTINGS["chat_ctx"]):
            mem["summary"] = ask(model, f"Existing summary:\n{mem['summary'] or '(none)'}\n\nNewer messages to fold in:\n"
                                    f"{transcript(chat[mem['upto']:cut], 500)}",
                             system="Update the running summary of this book-editing conversation in at most 150 "
                                    "words: requests made, decisions, changes applied, open questions. Plain text.")[:1200]
        mem["upto"] = cut
    except RuntimeError:
        pass


def chat_turn(book: Book, msg: str, model: str, emb: str, auto: bool) -> dict:
    mem, chat = st.session_state.memory, st.session_state.chat
    ctx = SETTINGS["chat_ctx"]
    predict = min(8192, max(1024, ctx // 4))      # room for the whole answer (plus hidden reasoning on gpt-oss)
    room = max(4000, int((ctx - predict - 2500) * 2.6))  # characters of context we can fill (≈2.6 chars/token, minus system prompt)
    prompt = (f"{notes_text() or 'AUTHOR PREFERENCES: (none yet)'}\n\n"
              f"EARLIER CONVERSATION (summary):\n{mem['summary'] or '(none)'}\n\n"
              f"RECENT CONVERSATION:\n{transcript(chat[-6:])}\n\n"
              f"BOOK OUTLINE:\n{book_outline(book, int(room * .3))}\n\n"
              f"EXCERPTS:\n{build_excerpts(book, msg, emb, int(room * .5), min(14, 4 + ctx // 8192)) or '(none)'}"
              f"\n\nUSER MESSAGE:\n{msg}")
    try:
        with use_ctx(ctx):
            r = ask(model, prompt, Reply, CHAT_SYS, temp=0.2, predict=predict)
    except RuntimeError as ex:
        return {"role": "assistant", "content": f"Sorry, {model} didn't give me a usable reply ({str(ex)[:140]}). "
                                                "Try again or pick another chat model.", "actions": []}
    add_notes(r.remember)
    ops, actions = [o for o in r.ops if o.op in OPS], []
    if ops and auto:
        actions = apply_ops(book, ops, model)
    elif ops:
        st.session_state.pending = [o.model_dump() for o in ops]
        actions = ["📝 Proposed (not applied yet): " + "; ".join(describe_op(o) for o in ops)]
    return {"role": "assistant", "content": r.answer.strip() or ("Done." if actions else "I have nothing to add."),
            "actions": actions}


# ───────────────────────── UI: tabs ─────────────────────────
def edit_tab(book: Book):
    rev, bp = st.session_state.rev, book.blueprint
    dirty = has_drafts(book)
    apply_bar("top", dirty)
    c1, c2 = st.columns(2)
    c1.text_input("Title", bp.title, key=f"e_title_{rev}")
    c2.text_input("Subtitle", bp.subtitle, key=f"e_sub_{rev}")
    c3, c4 = st.columns(2)
    c3.text_input("Author", book.author, key=f"e_author_{rev}")
    c4.selectbox("Subheading numbering", ["auto", "none", "keep"], index=["auto", "none", "keep"].index(bp.numbering)
                 if bp.numbering in ("keep", "auto", "none") else 1, key=f"e_num_{rev}",
                 format_func={"auto": "Auto-number (1.1, 1.2…) - consistent", "none": "No numbers",
                              "keep": "Keep numbers as written in the source"}.get,
                 help="Applies to all chapters when you click Apply changes. 'Keep' only shows numbers if the "
                      "text still contains them (books generated with auto/none numbering had them removed).")
    with st.expander("Preface", expanded=bool(book.preface.strip())):
        st.text_area("Preface (Markdown; leave empty for no preface)", book.preface, height=240, key=f"e_pref_{rev}")
    st.markdown("#### Chapters")
    for i, c in enumerate(book.chapters):
        with st.expander(f"Chapter {i + 1}: {c.title}"):
            st.text_input("Chapter title", c.title, key=f"e_ct_{c.id}_{rev}")
            st.text_area("Text (Markdown: `##` for subheadings, `**bold**`, lists, tables)", c.body, height=420,
                         key=f"e_cb_{c.id}_{rev}")
            b1, b2, b3, _ = st.columns([1, 1, 1, 6])
            b1.button("↑", key=f"up_{c.id}_{rev}", on_click=cb_move, args=(i, i - 1), disabled=i == 0, help="Move up")
            b2.button("↓", key=f"dn_{c.id}_{rev}", on_click=cb_move, args=(i, i + 1),
                      disabled=i == len(book.chapters) - 1, help="Move down")
            b3.button("🗑", key=f"del_{c.id}_{rev}", on_click=cb_delete, args=(i,),
                      disabled=len(book.chapters) < 2, help="Delete chapter (undo is available)")
    st.button("➕ Add chapter", on_click=cb_add_chapter)
    st.markdown("#### Glossary")
    gk = f"gl_src_{rev}"
    if gk not in st.session_state:  # constant source frame per revision; the editor keeps its own edits
        st.session_state[gk] = pd.DataFrame([{"term": t.term, "definition": t.definition} for t in book.glossary],
                                            columns=["term", "definition"])
    ed = st.data_editor(st.session_state[gk], num_rows="dynamic", key=f"e_gl_{rev}", width="stretch")

    def cell(x):
        return "" if x is None or (isinstance(x, float) and x != x) else str(x).strip()
    st.session_state[f"gl_cur_{rev}"] = [Term(term=cell(r.term), definition=cell(r.definition))
                                         for r in ed.itertuples(index=False) if cell(r.term)]
    st.divider()
    apply_bar("bottom", dirty)


def apply_bar(where: str, dirty: bool):
    a, b, c = st.columns([1.3, 1.3, 5])
    a.button("✅ Apply changes", type="primary", disabled=not dirty, on_click=cb_apply, key=f"apply_{where}",
             width="stretch")
    b.button("Discard edits", disabled=not dirty, on_click=cb_discard, key=f"discard_{where}", width="stretch")
    c.markdown("🟠 **Unapplied edits.** Click *Apply changes* to update the book (preview, exports and assistant)."
               if dirty else "🟢 All changes applied. Edit anything below, then click *Apply changes*.")


def contents_md(book: Book) -> str:
    lines = []
    for title, md in book.sections():
        lines.append(f"- **{title}**")
        if title.startswith("Chapter"):
            lines += [f"    - {h}" for h in re.findall(r"^##\s+(.+)$", md, re.M)]
    return "\n".join(lines)


def preview_tab(book: Book):
    if has_drafts(book):
        st.info("You have unapplied edits in the ✏️ Edit tab. This preview shows the book without them until you "
                "click *Apply changes*.")
    bp, secs = book.blueprint, book.sections()
    total = sum(words(md) for _, md in secs)
    st.caption(f"{len(book.chapters)} chapters · {total:,} words · roughly {max(1, total // 280)} printed pages (6×9 in)")
    mode = st.radio("View", ["Reader view", "Print layout (PDF)"], horizontal=True, label_visibility="collapsed",
                    key="pv_mode")
    if mode == "Reader view":
        with st.container(border=True):
            sub = f"<h3><i>{html.escape(bp.subtitle)}</i></h3>" if bp.subtitle else ""
            st.markdown(f"<div style='text-align:center;padding:4rem 0'><h1>{html.escape(bp.title)}</h1>{sub}"
                        f"<p style='font-size:1.1rem'>{html.escape(book.author)}</p></div>", unsafe_allow_html=True)
            st.divider()
            st.markdown("## Contents")
            st.markdown(contents_md(book))
            for title, md in secs:
                st.divider()
                st.markdown(f"# {title}")
                st.markdown(md)
    else:
        try:
            data = cached_export(book, "pdf")
            if data is None:
                with st.spinner("Typesetting the print layout…"):
                    data = make_export(book, "pdf")
            b64 = base64.b64encode(data).decode()
            components.html(f'<iframe src="data:application/pdf;base64,{b64}" '
                            'style="width:100%;height:100%;border:0"></iframe>', height=800)
            st.caption("Blank? Some browsers block embedded PDFs. Download it from the Export tab instead.")
        except Exception as ex:
            st.error(f"Couldn't build the print layout: {ex}")


def chat_tab(book: Book, names: list[str], emb: str, default_model: str):
    chat_names = [n for n in names if not re.search(r"nimble|embed|mimic|minilm|bge", n, re.I)] or names
    c1, c2, c3 = st.columns([3, 3, 1])
    with c1:
        model = pick("Assistant model", chat_names, [default_model, "gpt-oss:20b", "gemma4:12b"], key="chat_model")
    with c2:
        SETTINGS["chat_ctx"] = st.select_slider(
            "Assistant context length (tokens)", [8192, 16384, 32768, 65536, 131072], value=SETTINGS["chat_ctx"],
            key="chat_ctx_sel", help="How much of the book and conversation the assistant can read at once, and the "
                                     "room it has for its answer. Raise it if replies are cut off or it misses "
                                     "details; lower it if the model runs out of memory.")
    auto = st.checkbox("Apply changes automatically", True, key="chat_auto",
                       help="Off: the assistant proposes changes and waits for your OK. Undo works either way.")
    if c3.button("🗑 Clear", help="Clear the conversation (keeps the saved preferences)"):
        st.session_state.chat, st.session_state.pending = [], []
        st.session_state.memory.update(summary="", upto=0)
        st.rerun()
    if has_drafts(book):
        st.info("You have unapplied edits in the ✏️ Edit tab. The assistant works on the applied book; they are "
                "kept and applied automatically before it changes anything.")
    mem = st.session_state.memory
    with st.expander(f"🧠 Assistant memory ({len(mem['notes'])} preference{'s' * (len(mem['notes']) != 1)})"):
        st.caption("Standing preferences and decisions the assistant keeps for this book, one per line. "
                   "It adds to this when you say things like “always use British spelling”. Edit freely.")
        txt = st.text_area("Preferences", "\n".join(mem["notes"]), height=110, key=f"mem_notes_{st.session_state.mrev}")
        mem["notes"] = [x.strip() for x in txt.splitlines() if x.strip()]
        if mem["summary"]:
            st.markdown("**Earlier conversation (summary)**")
            st.write(mem["summary"])
    if not st.session_state.chat:
        st.info("Ask about the book (“What is chapter 3 about?”, “Where do I explain X?”) or tell me what to change "
                "(“Make chapter 2 more formal”, “Rename chapter 4 to …”, “Add a chapter about …”, "
                "“Replace every ‘utilise’ with ‘use’”).")
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            for a in m.get("actions", []):
                st.markdown(f"- {a}")
    if st.session_state.pending:
        ops = [Op(**d) for d in st.session_state.pending]
        st.warning("Proposed changes are waiting for your OK:\n\n" + "\n".join(f"- {describe_op(o)}" for o in ops))
        a, b, _ = st.columns([1, 1, 4])
        if a.button("Apply", type="primary"):
            with st.spinner("Applying changes…"):
                res = apply_ops(book, ops, model)
            st.session_state.pending = []
            st.session_state.chat.append({"role": "assistant", "content": "Applied the proposed changes.", "actions": res})
            st.rerun()
        if b.button("Discard"):
            st.session_state.pending = []
            st.rerun()
    prompt = st.chat_input("Ask about the book or tell me what to change…", key="chat_in")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = chat_turn(book, prompt, model, emb, auto)
            st.markdown(reply["content"])
        st.session_state.chat += [{"role": "user", "content": prompt, "actions": []}, reply]
        roll_memory(model)
        st.rerun()


def export_tab(book: Book):
    name = re.sub(r"[^\w\-]+", "_", book.blueprint.title)[:60] or "book"
    st.caption("Exports always use the latest version of the book, including your edits and the assistant's changes.")
    c1, c2, c3, c4 = st.columns(4)
    for col, fmt, mime in ((c1, "pdf", "application/pdf"), (c2, "epub", "application/epub+zip")):
        data = cached_export(book, fmt)
        if data is not None:
            col.download_button(f"⬇ {fmt.upper()}", data, f"{name}.{fmt}", mime, width="stretch")
        elif col.button(f"Prepare {fmt.upper()}", width="stretch", key=f"prep_{fmt}"):
            with st.spinner(f"Building {fmt.upper()}…"):
                try:
                    make_export(book, fmt)
                except Exception as ex:
                    st.error(f"{fmt.upper()} export failed: {ex}")
                    with st.expander("Technical details (send this if the problem persists)"):
                        st.code(traceback.format_exc())
                    continue
            st.rerun()
    c3.download_button("⬇ Markdown", full_markdown(book), f"{name}.md", "text/markdown", width="stretch")
    c4.download_button("💾 Project", project_json(book), f"{name}.bookforge.json", "application/json",
                       width="stretch", help="Book + assistant memory. Re-open it later from 'Create or open'.")


def workspace(book: Book, names: list[str], emb: str, default_model: str):
    bp = book.blueprint
    h1, h2, h3 = st.columns([6, 1, 1.4])
    h1.subheader(f"📕 {bp.title}")
    h2.button("↩ Undo", on_click=cb_undo, disabled=not st.session_state.undo, key="undo_btn")
    h3.button("⟲ Revert all", on_click=cb_revert, key="revert_btn", help="Go back to the originally generated book")
    st.caption(f"{bp.genre} · {bp.tone} · {len(book.chapters)} chapters"
               + (f" · meaning fidelity {book.fidelity:.0%}" if book.fidelity else ""))
    if book.unpolished:
        st.warning(f"{book.unpolished} passage(s) could not be rewritten by the writer model and were kept as your "
                   "original text." + (f" First reason: {book.why}" if book.why else ""))
    if msg := st.session_state.pop("flash", None):
        st.toast(msg, icon="✅")
    t_prev, t_edit, t_chat, t_exp = st.tabs(["👁 Preview", "✏️ Edit", "💬 Assistant", "⬇ Export"])
    with t_edit:  # rendered first in code so the preview always sees the latest edits
        edit_tab(book)
    with t_prev:
        preview_tab(book)
    with t_chat:
        chat_tab(book, names, emb, default_model)
    with t_exp:
        export_tab(book)


# ───────────────────────── UI (single page) ─────────────────────────
def main():
    st.set_page_config(page_title="Book Forge", page_icon="📖", layout="wide")
    st.title("📖 Book Forge")
    st.caption("Any text in, a polished book out. Meaning, tone and subject preserved. 100% local via Ollama.")
    try:
        names = local_models()
    except Exception:
        st.error("Can't reach Ollama. Start it (`ollama serve`) and refresh.")
        st.stop()

    init_state()

    with st.sidebar:
        st.header("Models")
        decider = pick("Decision model", names, ["nimble:9b", "nimble"])
        writer = pick("Writer model", names, ["gpt-oss:20b", "gemma4:12b"])
        emb = pick("Embedding model", names, ["mimic", "nomic-embed-text"])
        verify = st.checkbox("Verify each passage with the decision model", value=True,
                             help="nimble double-checks that no meaning was lost. Slower but safer.")
        ctx = st.select_slider("Context length (tokens)", [4096, 8192, 16384, 32768, 65536, 131072],
                               value=SETTINGS["ctx"],
                               help="Context window for polishing and editing calls. Bigger holds more text but uses "
                                    "more VRAM and is slower. 8192 suits a 900-word passage. The assistant has its own "
                                    "setting in its tab.")
        SETTINGS["ctx"] = ctx
        workers = st.slider("Parallel passages", 1, 4, 2,
                            help="Faster on a GPU with spare memory. For a real speed-up start Ollama with "
                                 "OLLAMA_NUM_PARALLEL set to the same number; otherwise passages just queue.")
        st.header("Optional")
        title = st.text_input("Book title (blank = AI decides)")
        author = st.text_input("Author name")

    has_book = st.session_state.get("book") is not None
    created = False
    with st.expander("➕ Create or open a book", expanded=not has_book):
        files = st.file_uploader("Upload PDF, DOCX, TXT or MD", type=["pdf", "docx", "doc", "txt", "md"],
                                 accept_multiple_files=True)
        pasted = st.text_area("…or paste anything: notes, scribbles, drafts", height=200)

        if st.button("Publish my book", type="primary", disabled=not (files or pasted.strip())):
            bar = None
            try:
                text = "\n\n".join([extract_text(f.name, f.getvalue()) for f in files] + [pasted]).strip()
                if len(text.split()) < 5:
                    raise ValueError("Not enough text to build a book from.")
                bar = st.progress(0.0, "Starting…")
                set_book(make_book(text, decider, writer, emb, title.strip(), author.strip(), bar, verify, workers))
                created = True
            except Exception as ex:
                if bar:
                    bar.empty()
                st.error(str(ex))
        st.markdown("**Or continue a saved project**")
        proj = st.file_uploader("Project file (.bookforge.json)", type=["json"], key="proj_up")
        if proj and st.button("Open project"):
            try:
                d = json.loads(proj.getvalue())
                set_book(Book.model_validate(d["book"]))
                st.session_state.chat = d.get("chat", [])
                st.session_state.memory = {**new_memory(), **d.get("memory", {})}
                created = True
            except Exception as ex:
                st.error(f"That doesn't look like a Book Forge project: {str(ex)[:200]}")
    if created:
        st.rerun()

    if book := st.session_state.get("book"):
        workspace(book, names, emb, writer)


if __name__ == "__main__":
    main()

"""One-off script: add YAML frontmatter (title + description) and infer code-
fence language tags across all docs markdown files. Idempotent — re-runs are
no-ops once every file is normalised.

Usage::

    uv run python scripts/llmify_docs.py [--check]

``--check`` exits 1 if any file would change without writing — wire into CI
once everything is clean.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"

# Order matters for fence inference: more-specific patterns first. The keys are
# (lowercase) substrings looked for in the fence body; the value is the language
# tag we set when no explicit tag is present.
FENCE_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\s*defaults\s*:", re.MULTILINE), "yaml"),
    (re.compile(r"^\s*hydra\s*:", re.MULTILINE), "yaml"),
    (re.compile(r"^\s*_target_\s*:", re.MULTILINE), "yaml"),
    (
        re.compile(r"^\s*(transparency|robustness|metrics|reporting|tracking)\s*:", re.MULTILINE),
        "yaml",
    ),
    (re.compile(r"^from\s+\w[\w.]*\s+import\b", re.MULTILINE), "python"),
    (re.compile(r"^import\s+\w", re.MULTILINE), "python"),
    (re.compile(r"^\s*(def|class|@\w)", re.MULTILINE), "python"),
    (re.compile(r"^\$\s|^\s*(uv|pip|raitap|gh|git|cd|python|pytest)\s", re.MULTILINE), "bash"),
    (re.compile(r"^\s*\{", re.MULTILINE), "json"),
)


def infer_fence_lang(body: str) -> str | None:
    for pattern, lang in FENCE_HINTS:
        if pattern.search(body):
            return lang
    return None


FENCE_RE = re.compile(r"(?ms)^(```)([^\n`]*)\n(.*?)^```$")


def fix_fences(text: str) -> tuple[str, int]:
    fixed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal fixed
        opening, info, body = match.group(1), match.group(2).strip(), match.group(3)
        if info:
            return match.group(0)
        lang = infer_fence_lang(body)
        if not lang:
            return match.group(0)
        fixed += 1
        return f"{opening}{lang}\n{body}```"

    new = FENCE_RE.sub(repl, text)
    return new, fixed


H1_RE = re.compile(r"^# (.+?)\s*$", re.MULTILINE)
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


DIRECTIVE_BLOCK_RE = re.compile(r"```\{[^\n}]+\}.*?^```$", re.DOTALL | re.MULTILINE)
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
# ``config-page`` directives carry an ``:intro:`` field with a one-paragraph
# overview written for humans — perfect for the ``description`` slot. The intro
# can wrap across lines until the next ``:option:`` / ``:yaml:`` / ``:python:``
# / ``:cli:`` field marker.
INTRO_RE = re.compile(r"^:intro:\s*(.+?)(?=^:\w+:|^```$)", re.MULTILINE | re.DOTALL)
SUMMARY_RE = re.compile(r"^:summary:\s*(.+?)(?=^:\w+:|^```$)", re.MULTILINE | re.DOTALL)


def directive_intro(text: str) -> str:
    """Pull a description out of a ``{config-page}`` ``:intro:`` or a
    ``{recipe}`` ``:summary:`` field — both serve the "one-paragraph
    overview" role even though they live inside MyST directive blocks that
    :func:`first_paragraph` deliberately strips.
    """
    match = INTRO_RE.search(text) or SUMMARY_RE.search(text)
    if not match:
        return ""
    cleaned = " ".join(match.group(1).split())
    cleaned = re.sub(r"[`*]", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    return _truncate_at_word_boundary(cleaned, 240)


def first_paragraph(text: str) -> str:
    """Pick the first prose paragraph after the H1 to seed ``description``.

    Skips frontmatter, code blocks (regular + MyST directive-flavoured), lists,
    tables, directives, and anything that looks more like code than English.
    """
    after_h1 = H1_RE.split(text, maxsplit=1)
    body = after_h1[-1] if len(after_h1) > 1 else text
    # Strip out every fenced block — regular ``` … ``` and ```{directive} … ```
    # so we never sample text from inside a code/directive body.
    body = DIRECTIVE_BLOCK_RE.sub("", body)
    body = CODE_BLOCK_RE.sub("", body)
    for raw in body.split("\n\n"):
        para = raw.strip()
        if not para:
            continue
        first = para.splitlines()[0]
        # Skip non-prose openings: headings, lists, tables, directive marker
        # lines (``:option:``), html, blockquotes, definition entries.
        if first.startswith(("#", "-", "*", "|", "{", ":", "<", "!", ">", "    ", "\t")):
            continue
        # Strip markdown emphasis (backticks + bold/italic asterisks) but keep
        # underscores — they're part of identifiers like ``vit_b_32`` that read
        # naturally in prose descriptions.
        cleaned = re.sub(r"[`*]", "", " ".join(para.split()))
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        # Bold markers like ``**word**`` left a leading asterisk-stripped form
        # behind; remove paired underscores used as emphasis only when they
        # surround a single word (``_word_``).
        cleaned = re.sub(r"\b_(\w+)_\b", r"\1", cleaned)
        # Heuristic: reject paragraphs that look like code rather than prose
        # (lots of ``=`` / ``(`` / ``)`` and few spaces between sentences).
        words = cleaned.split()
        if len(words) < 6:
            continue
        punctuation_density = sum(cleaned.count(ch) for ch in "(){}=[]") / max(len(cleaned), 1)
        if punctuation_density > 0.05:
            continue
        return _truncate_at_word_boundary(cleaned, 240)
    return ""


def _truncate_at_word_boundary(text: str, limit: int) -> str:
    """Cap at ``limit`` chars without breaking a word; prefer a sentence end."""
    if len(text) <= limit:
        return text.rstrip()
    cut = text[:limit]
    # Prefer the last sentence boundary in range; fall back to a space.
    sentence_break = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if sentence_break >= 80:
        return cut[: sentence_break + 1].rstrip()
    space = cut.rfind(" ")
    if space >= 80:
        return cut[:space].rstrip() + "…"
    return cut.rstrip() + "…"


DESC_LINE_RE = re.compile(r'^description:\s*"(.*)"\s*$', re.MULTILINE)


def looks_like_bad_description(desc: str) -> bool:
    """Heuristic for previously-mis-generated descriptions that grabbed a code
    sample instead of prose. High punctuation density, obvious assignment
    syntax, or mid-word truncation triggers a rewrite on the next pass."""
    if not desc:
        return True
    if "=" in desc and any(token in desc for token in ("(", ")")):
        return True
    punct = sum(desc.count(ch) for ch in "(){}=[]") / max(len(desc), 1)
    if punct > 0.05:
        return True
    # Mid-word truncation (an older script stripped 240 chars without honouring
    # word boundaries — the tail looks like a half-cut word).
    if desc.endswith(("the", "a", "an", "of", "in", "to", "for")):
        return True
    last_word = desc.rsplit(" ", 1)[-1] if " " in desc else desc
    return bool(
        last_word and last_word[-1].isalpha() and len(last_word) < 4 and not desc.endswith(".")
    )


TOP_LEVEL_HTML_META_RE = re.compile(r"^html_meta:\n((?:[ \t]+.+\n)+)", re.MULTILINE)


def migrate_top_level_html_meta(block: str) -> str:
    """Move legacy top-level ``html_meta:`` under the ``myst:`` key.

    MyST deprecated the top-level form in 5.x — leaving it triggers build
    warnings. If a ``myst:`` block already exists the top-level copy is just
    dropped (the nested version is the source of truth). Idempotent.
    """
    match = TOP_LEVEL_HTML_META_RE.search(block)
    if not match:
        return block
    if "myst:" in block:
        # Nested form already present; just strip the deprecated duplicate.
        return block.replace(match.group(0), "", 1)
    body = match.group(1)
    indented = "".join("  " + line for line in body.splitlines(keepends=True))
    replacement = "myst:\n  html_meta:\n" + indented
    return block.replace(match.group(0), replacement, 1)


def ensure_frontmatter(text: str, fallback_title: str) -> tuple[str, bool]:
    existing = FRONTMATTER_RE.match(text)
    if existing:
        migrated = migrate_top_level_html_meta(existing.group(1))
        if migrated != existing.group(1):
            text = text.replace(existing.group(0), f"---\n{migrated}\n---\n", 1)
            existing = FRONTMATTER_RE.match(text)
    # Strip code/directive blocks before scanning for H1 so we never grab a
    # Python comment like ``# Option A …`` that lives inside a fenced snippet.
    prose_only = CODE_BLOCK_RE.sub("", DIRECTIVE_BLOCK_RE.sub("", text))
    h1_match = H1_RE.search(prose_only)
    title = h1_match.group(1).strip() if h1_match else fallback_title
    title = re.sub(r"`", "", title).strip()
    # Don't accept a "title" pulled from a Python comment inside a code block —
    # MyST H1s never end with a colon / open paren / unfinished clause.
    if title.endswith((":", "(", ",", ";")) or not title:
        title = fallback_title
    description = first_paragraph(text) or directive_intro(text)

    if existing:
        block = existing.group(1)
        title_match = re.search(r'^title:\s*"(.*)"\s*$', block, re.MULTILINE)
        current_title = title_match.group(1) if title_match else ""
        bad_title = title_match is not None and current_title.endswith((":", "(", ",", ";"))
        needs_title = title_match is None or bad_title
        current_desc_match = DESC_LINE_RE.search(block)
        current_desc = current_desc_match.group(1) if current_desc_match else ""
        bad_desc = current_desc_match is not None and looks_like_bad_description(current_desc)
        needs_desc = (current_desc_match is None and bool(description)) or bad_desc
        has_html_meta = "html_meta:" in block and "myst:" in block
        final_desc = description if (needs_desc or not current_desc) else current_desc
        needs_html_meta = not has_html_meta and bool(final_desc)
        if not (needs_title or needs_desc or needs_html_meta):
            return text, False
        lines = block.splitlines()
        if bad_title and title:
            lines = [
                line if not line.startswith("title:") else f'title: "{title}"' for line in lines
            ]
        if bad_desc and description:
            lines = [
                line if not line.startswith("description:") else f'description: "{description}"'
                for line in lines
            ]
        if needs_title and not bad_title and title:
            lines.append(f'title: "{title}"')
        if needs_desc and not bad_desc:
            lines.append(f'description: "{description}"')
        if needs_html_meta:
            lines.append("myst:")
            lines.append("  html_meta:")
            lines.append(f'    "description": "{final_desc}"')
        new_block = "\n".join(lines)
        return text.replace(existing.group(0), f"---\n{new_block}\n---\n", 1), True

    parts = ["---"]
    if title:
        parts.append(f'title: "{title}"')
    if description:
        # ``description:`` is the plain, LLM-readable copy. The ``myst.html_meta``
        # block is what MyST turns into a real ``<meta>`` tag in the rendered
        # HTML head (Furo + sphinx-sitemap pick it up).
        parts.append(f'description: "{description}"')
        parts.append("myst:")
        parts.append("  html_meta:")
        parts.append(f'    "description": "{description}"')
    parts.append("---\n")
    return "\n".join(parts) + "\n" + text, True


def process(path: Path, *, check: bool) -> bool:
    original = path.read_text(encoding="utf-8")
    text, _ = fix_fences(original)
    text, _ = ensure_frontmatter(text, path.stem.replace("-", " ").title())
    if text == original:
        return False
    if check:
        return True
    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    changed: list[Path] = []
    for md in sorted(DOCS.rglob("*.md")):
        if "_build" in md.parts or "archive" in md.parts or "superpowers" in md.parts:
            continue
        if process(md, check=args.check):
            changed.append(md)

    if not changed:
        print("All docs already llm-normalised.")
        return 0

    label = "would change" if args.check else "updated"
    print(f"{label} ({len(changed)} files):")
    for path in changed:
        print(f"  {path.relative_to(DOCS.parent)}")
    return 1 if args.check else 0


if __name__ == "__main__":
    sys.exit(main())

"""Render dogfood_corpus_diverse.py into a clean, copy-paste-by-hand markdown
walkthrough for pasting turn-by-turn into a LIVE OpenCode session (Qwen).

Why this exists alongside run_local_dogfood.py: that script drives the corpus
unattended over HTTP using the scripted assistant replies baked into the
corpus. This script is for doing the same conversations BY HAND instead, so
you see Qwen's real replies, watch the graph fill in live via `/memory`, and
can judge extraction/merge/retrieval quality turn by turn rather than from a
batch report after the fact.

The corpus is the single source of truth (dogfood_corpus_diverse.py) -- this
script only renders it. Regenerate after editing the corpus:

    uv run python scripts/render_corpus_markdown.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dogfood_corpus_diverse import CONVERSATIONS, PROBES  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "dogfood_corpus_manual.md"

HEADER = """# Dogfood corpus — manual walkthrough (Qwen / OpenCode)

Paste this in by hand, turn by turn, into a live `qwen-code` session, so you
can read Qwen's real replies and watch `/memory` fill in as you go.

## How to run this

1. **Start a brand-new OpenCode session before EACH numbered conversation
   below, and exit OpenCode (so the session actually closes) once you finish
   a conversation's turns.** This isn't just tidiness -- it matters for two
   real reasons:
   - The write hook only fires once per session, at exit (`dispose`) -- a
     conversation isn't consolidated into the graph until you actually quit.
   - Several probes at the end are specifically testing whether the **graph**
     resurfaces a fact appropriately. If everything ran in one long session,
     Qwen's own live chat context (not the graph) would explain any recall,
     and the test would stop measuring what it's meant to measure.
2. **Only paste the bold `Turn` blocks** (in code fences, so they're easy to
   select and copy cleanly). The italicized blockquote under each turn is
   just a *reference* of what the scripted version of this conversation
   assumed the assistant might say -- it's there so you know the intended
   direction, not something to paste. Qwen's actual reply will differ in
   wording; that's fine, just read it and move to the next turn regardless of
   exactly how it phrased things.
3. Run the conversations **in the numbered order** -- this is one persona's
   memory graph accumulating over weeks, and several later conversations
   (job change, corrections, callbacks) assume earlier ones already landed.
4. After all conversations are done and each session has been closed (so
   every write has actually fired), move to the **Probes** section at the
   bottom. Same rule applies: **each probe should be asked in its own fresh
   session** -- the point is to see what the graph alone surfaces with zero
   live chat history to lean on.
5. Check `/memory` (or `http://127.0.0.1:8765/viewer`) whenever you want to
   see the graph directly instead of only through what gets injected.

---
"""

PROBES_HEADER = """
---

## Probes — run each in a brand-new session

For each one: start a fresh OpenCode session, paste only the question, and
compare what actually gets injected/said against the `expect` note.
"""


def render_conversation(i: int, title: str, turns: list[tuple[str, str]]) -> str:
    lines = [f"## {i}. `{title}`", ""]
    turn_num = 0
    for role, text in turns:
        if role == "user":
            turn_num += 1
            lines.append(f"**Turn {turn_num}:**")
            lines.append("```")
            lines.append(text)
            lines.append("```")
        else:
            lines.append(f"> *(reference reply, don't paste — Qwen's real answer will differ)*")
            lines.append(f"> {text}")
        lines.append("")
    return "\n".join(lines)


def render_probe(i: int, query: str, expect: str) -> str:
    return (
        f"### Probe {i}\n\n"
        f"```\n{query}\n```\n\n"
        f"*expect*: {expect}\n"
    )


def main() -> None:
    parts = [HEADER]
    for i, (title, turns) in enumerate(CONVERSATIONS, 1):
        parts.append(render_conversation(i, title, turns))
        parts.append("---\n")
    parts.append(PROBES_HEADER)
    for i, (query, expect) in enumerate(PROBES, 1):
        parts.append(render_probe(i, query, expect))
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(CONVERSATIONS)} conversations, {len(PROBES)} probes)")


if __name__ == "__main__":
    main()

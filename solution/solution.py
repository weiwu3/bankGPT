"""Entry point for the scorer: build() returns an object with ingest, answer and conflicts.

The work is done in the `kb` package next to this file; NOTES.md explains the approach.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:                                    # the scorer puts its own contract module on the path
    from contract import Answer, Conflict, Source
except ImportError:                     # run on its own: same shapes, defined locally
    from dataclasses import dataclass, field

    @dataclass
    class Source:
        url: str
        snapshot_id: Optional[str] = None
        quote: Optional[str] = None

    @dataclass
    class Answer:
        text: str
        sources: list = field(default_factory=list)
        confident: bool = True
        conflict_note: Optional[str] = None

    @dataclass
    class Conflict:
        topic: str
        values: list
        sources: list
        resolution: Optional[str] = None
        reason: Optional[str] = None

from kb.engine import Engine           # noqa: E402
from kb.reader import make_reader       # noqa: E402
from kb.reply import refusal            # noqa: E402


class KnowledgeSystem:
    """The contract's object: kb's Engine behind the scorer's Answer, Source and Conflict shapes."""

    def __init__(self) -> None:
        self.engine = Engine(reader=make_reader())

    def ingest(self, snapshot) -> None:
        """Read one snapshot; they arrive oldest first."""
        self.engine.ingest(snapshot)

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        """The answer with its sources, for the site as of `as_of` (the newest download when None)."""
        try:
            reply = self.engine.answer(str(question or ""), as_of)
            view = self.engine.store.view_at(as_of)
        except Exception:               # a question that breaks the code gets an honest "I don't know", not a crash
            traceback.print_exc()
            return Answer(text=refusal().text, confident=False)
        sources = [Source(url=p.url, snapshot_id=view.snapshot_id if view else None, quote=reply.quotes.get(p.key))
                   for p in reply.pages]
        return Answer(text=reply.text, sources=sources, confident=reply.confident, conflict_note=reply.note)

    def conflicts(self) -> list:
        """Every contradiction across the downloads, with the value that stands and why."""
        return [Conflict(topic=c.topic, values=c.values,
                         sources=[Source(url=u, snapshot_id=s, quote=q) for u, s, q in c.sources],
                         resolution=c.resolution, reason=c.reason)
                for c in self.engine.conflicts()]


def build() -> KnowledgeSystem:
    """A new, empty knowledge system."""
    return KnowledgeSystem()

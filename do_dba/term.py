"""Terminal encoding safety.

On Windows a redirected stdout defaults to the locale code page (e.g. cp1252),
which cannot encode either our decorative glyphs or ordinary model output like
emoji and CJK. Left alone that raises UnicodeEncodeError mid-reply, so we widen
the streams where we can and fall back to ASCII glyphs where we cannot.
"""

from __future__ import annotations

import sys

_FANCY = "↳·✓…"


def prepare_streams() -> None:
    """Make stdout/stderr tolerant of any character a model might emit."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if stream.isatty():
                # A real console already knows its own encoding; only make it
                # non-fatal so an unmappable character cannot kill the stream.
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def supports_fancy_glyphs() -> bool:
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if not encoding:
        return False
    try:
        _FANCY.encode(encoding, errors="strict")
    except (LookupError, UnicodeEncodeError):
        return False
    return True


class Glyphs:
    """Decorative characters, with ASCII stand-ins for narrow encodings."""

    def __init__(self, fancy: bool):
        self.fancy = fancy
        self.reply = "↳" if fancy else "->"
        self.sep = "·" if fancy else "|"
        self.check = "✓" if fancy else "+"
        self.cont = "…" if fancy else "..."

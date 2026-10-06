"""Arabic typed with the keyboard left on English: «hgvfh» is «الربا» (h=ا g=ل v=ر f=ب h=ا).

The standard Arabic layout, unshifted keys only. Nothing is guessed from meaning: a message is read as Arabic only
when it is Latin letters alone and every word of the result is found in the sources (pipeline._respond), so an
English question stays English.
"""
from __future__ import annotations

import re

LAYOUT = {
    "q": "ض", "w": "ص", "e": "ث", "r": "ق", "t": "ف", "y": "غ", "u": "ع", "i": "ه", "o": "خ", "p": "ح",
    "[": "ج", "]": "د", "a": "ش", "s": "س", "d": "ي", "f": "ب", "g": "ل", "h": "ا", "j": "ت", "k": "ن",
    "l": "م", ";": "ك", "'": "ط", "z": "ئ", "x": "ء", "c": "ؤ", "v": "ر", "b": "لا", "n": "ى", "m": "ة",
    ",": "و", ".": "ز", "/": "ظ", "`": "ذ",
}
_LATIN_ONLY = re.compile(r"^[A-Za-z\[\];',./`\s]+$")


def latin_only(text: str) -> bool:
    """Only keys of the Latin layout, with at least three letters."""
    return bool(_LATIN_ONLY.match(text or "")) and sum(c.isalpha() for c in text) >= 3


def to_arabic(text: str) -> str:
    """The same keys read on the Arabic layout (a question mark typed as «?» stays as it is)."""
    return "".join(LAYOUT.get(c.lower(), c) for c in text)

"""Where to turn when Muhawir refers the user on: named, official bodies with their own websites.

Muhawir does not answer a personal case or a contemporary financial ruling itself. Instead of
only saying «ask a specialist», each referred reply carries a card that names who to ask, by kind of case:

  fatwa      a personal case, a ruling, or a question the sources do not answer: official fatwa bodies
  finance    contemporary financial matters: the official fatwa body
  newcomer   someone who wants to become Muslim: a mosque or Islamic centre near them, people first

Every link is the body's own official website, checked when it was added (4 October 2026). The list is
limited to Saudi bodies (the project owner's choice); the user is always told first to ask the official body of
their own country. Phone numbers are given only where the body's own official page states them.
"""
from __future__ import annotations

import re

from .normalize import normalize

FATWA_BODIES = [
    {"ar": "الرئاسة العامة للبحوث العلمية والإفتاء", "en": "General Presidency of Scholarly Research and Ifta",
     "where_ar": "السعودية", "where_en": "Saudi Arabia", "url": "https://www.alifta.gov.sa"},
]

CARD = {
    "ar": {
        "fatwa": ("من تسأل؟", "اسأل جهة الفتوى الرسمية في بلدك. وفي السعودية:"),
        "finance": ("من تسأل؟", "القضايا المالية المعاصرة تبحثها جهات الفتوى الرسمية والمجامع الفقهية. اسأل جهة الفتوى الرسمية في بلدك. وفي السعودية:"),
        "newcomer": ("تحدّث مع أحد", "تواصل مع أقرب مسجد أو مركز إسلامي إليك، فهم يرحبون بك ويساعدونك خطوة خطوة، "
                                    "ويجيبون عن أسئلتك وجهًا لوجه."),
    },
    "en": {
        "fatwa": ("Who to ask", "Ask the official fatwa body in your country. In Saudi Arabia:"),
        "finance": ("Who to ask", "Contemporary financial matters are studied by official fatwa bodies and fiqh academies. "
                                  "Ask the official fatwa body in your country. In Saudi Arabia:"),
        "newcomer": ("Talk to someone", "Contact the mosque or Islamic centre nearest to you: they will welcome you, help you "
                                       "step by step, and answer your questions in person."),
    },
}

# someone who wants to become Muslim (not a question about conversion in general)
_NEWCOMER = re.compile(
    r"\b(?:how (?:do|can|to) (?:i|you) (?:become|be) (?:a )?muslim|(?:i want|i'd like|i would like) to (?:become|be) "
    r"(?:a )?muslim|(?:convert|revert)(?:ing)? to islam|embrace islam)\b"
    # «كيف أسلم؟» (how do I become Muslim) but not «كيف أسلم عمر؟» (how did Umar become Muslim)
    r"|(?:كيف|اريد ان|ابغي|ابغى|ابي|اود ان|احب ان)\s+(?:اسلم(?=\s*[؟?.!]?\s*$)|ادخل (?:في )?الاسلام|اعتنق الاسلام)",
    re.IGNORECASE)


def wants_to_become_muslim(question: str) -> bool:
    return bool(_NEWCOMER.search(question or "") or _NEWCOMER.search(normalize(question or "")))


def card(kind: str, lang: str) -> dict:
    """The referral card for one kind of case, in the reply's language."""
    lang = lang if lang in CARD else "ar"
    title, intro = CARD[lang][kind]
    bodies = {"fatwa": FATWA_BODIES, "finance": FATWA_BODIES,
              "newcomer": []}[kind]
    links = [{"label": b[lang], "where": b[f"where_{lang}"], "url": b["url"]} for b in bodies]
    return {"kind": kind, "title": title, "intro": intro, "links": links}

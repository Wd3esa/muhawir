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
     "ur": "الرئاسة العامة للبحوث العلمية والإفتاء (ادارۂ تحقیقاتِ علمیہ و افتاء)",
     "where_ar": "المملكة العربية السعودية", "where_en": "Saudi Arabia", "where_ur": "سعودی عرب",
     "url": "https://www.alifta.gov.sa"},
]

CARD = {
    "ar": {
        "fatwa": ("من تسأل؟", "جهة الفتوى الرسمية في بلدك، مثل:"),
        "finance": ("من تسأل؟", "جهة الفتوى الرسمية في بلدك، مثل:"),
        "newcomer": ("تحدّث مع أحد", "تواصل مع أقرب مسجد أو مركز إسلامي إليك، فهم يرحبون بك ويساعدونك خطوة خطوة، "
                                    "ويجيبون عن أسئلتك وجهًا لوجه."),
    },
    "en": {
        "fatwa": ("Who to ask", "The official fatwa body in your country, for example:"),
        "finance": ("Who to ask", "The official fatwa body in your country, for example:"),
        "newcomer": ("Talk to someone", "Contact the mosque or Islamic centre nearest to you: they will welcome you, help you "
                                       "step by step, and answer your questions in person."),
    },
    "ur": {
        "fatwa": ("کس سے پوچھیں؟", "آپ کے ملک کا سرکاری دارالافتاء، جیسے:"),
        "finance": ("کس سے پوچھیں؟", "آپ کے ملک کا سرکاری دارالافتاء، جیسے:"),
        "newcomer": ("کسی سے بات کریں", "اپنے قریب ترین مسجد یا اسلامی مرکز سے رابطہ کریں: وہ آپ کو خوش آمدید کہیں گے، قدم بہ قدم "
                                         "آپ کی مدد کریں گے اور آمنے سامنے آپ کے سوالوں کے جواب دیں گے۔"),
    },
}

# someone who wants to become Muslim (not a question about conversion in general)
_NEWCOMER = re.compile(
    r"\b(?:how (?:do|can|to) (?:i|you) (?:become|be) (?:a )?muslim|(?:i want|i'd like|i would like) to (?:become|be) "
    r"(?:a )?muslim|(?:convert|revert)(?:ing)? to islam|embrace islam)\b"
    # «كيف أسلم؟» (how do I become Muslim) but not «كيف أسلم عمر؟» (how did Umar become Muslim)
    r"|(?:كيف|اريد ان|ابغي|ابغى|ابي|اود ان|احب ان)\s+(?:اسلم(?=\s*[؟?.!]?\s*$)|ادخل (?:في )?الاسلام|اعتنق الاسلام)"
    # Urdu: «میں مسلمان ہونا چاہتا ہوں»، «اسلام قبول کرنا چاہتی ہوں»
    r"|مسلمان (?:ہونا|بننا) چا[ہه]ت[اےی]|اسلام قبول [کك]رنا چا[ہه]ت[اےی]",
    re.IGNORECASE)


def wants_to_become_muslim(question: str) -> bool:
    """True when the message says the person wants to become a Muslim."""
    return bool(_NEWCOMER.search(question or "") or _NEWCOMER.search(normalize(question or "")))


def card(kind: str, lang: str) -> dict:
    """The referral card for one kind of case, in the reply's language."""
    lang = lang if lang in CARD else "ar"
    title, intro = CARD[lang][kind]
    bodies = {"fatwa": FATWA_BODIES, "finance": FATWA_BODIES,
              "newcomer": []}[kind]
    links = [{"label": b[lang], "where": b[f"where_{lang}"], "url": b["url"]} for b in bodies]
    return {"kind": kind, "title": title, "intro": intro, "links": links}

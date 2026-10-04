"""Where to turn when Muhawir refers the user on: named, official bodies with their own websites.

Muhawir does not answer a personal case, a contemporary financial ruling or a crisis itself. Instead of
only saying «ask a specialist», each referred reply carries a card that names who to ask, by kind of case:

  fatwa      a personal case, a ruling, or a question the sources do not answer: official fatwa bodies
  finance    contemporary financial matters: the fiqh academy first, then the official fatwa bodies
  crisis     thoughts of self-harm or danger: emergency services and support lines, people first
  newcomer   someone who wants to become Muslim: a mosque or Islamic centre near them, people first

Every link is the body's own official website, checked when it was added (4 October 2026). The list is
deliberately short and is not a ranking: the user is always told to prefer the official body of their own
country. Phone numbers are given only where the body's own official page states them.
"""
from __future__ import annotations

import re

from .normalize import normalize

FATWA_BODIES = [
    {"ar": "الرئاسة العامة للبحوث العلمية والإفتاء", "en": "General Presidency of Scholarly Research and Ifta",
     "where_ar": "السعودية", "where_en": "Saudi Arabia", "url": "https://www.alifta.gov.sa"},
    {"ar": "دار الإفتاء المصرية", "en": "Dar al-Ifta al-Misriyyah",
     "where_ar": "مصر", "where_en": "Egypt", "url": "https://www.dar-alifta.org"},
    {"ar": "دائرة الإفتاء العام", "en": "General Iftaa' Department",
     "where_ar": "الأردن", "where_en": "Jordan", "url": "https://www.aliftaa.jo"},
    {"ar": "الهيئة العامة للشؤون الإسلامية والأوقاف", "en": "General Authority of Islamic Affairs and Endowments",
     "where_ar": "الإمارات", "where_en": "UAE", "url": "https://www.awqaf.gov.ae"},
]

FIQH_ACADEMY = {"ar": "مجمع الفقه الإسلامي الدولي (منظمة التعاون الإسلامي)",
                "en": "International Islamic Fiqh Academy (Organisation of Islamic Cooperation)",
                "where_ar": "قرارات المجمع في القضايا المعاصرة", "where_en": "its resolutions on contemporary issues",
                "url": "https://www.iifa-aifi.org"}

# the Saudi Ministry of Health page lists both numbers (moh.gov.sa, «الصحة النفسية»)
CRISIS_LINES = [
    {"ar": "مركز الاستشارات النفسية بوزارة الصحة: 920033360", "en": "Ministry of Health psychological consultation centre: 920033360",
     "where_ar": "السعودية، من 8 صباحًا إلى 8 مساءً", "where_en": "Saudi Arabia, 8 am to 8 pm",
     "url": "https://www.moh.gov.sa/Ministry/Information-and-services/Pages/psychiatry.aspx"},
    {"ar": "Find A Helpline: خطوط دعم مجانية موثقة", "en": "Find A Helpline: free, verified support lines",
     "where_ar": "أكثر من 175 دولة", "where_en": "175+ countries", "url": "https://findahelpline.com"},
]

CARD = {
    "ar": {
        "fatwa": ("من تسأل؟", "اسأل جهة الفتوى الرسمية في بلدك، ومنها:"),
        "finance": ("من تسأل؟", "القضايا المالية المعاصرة تبحثها المجامع الفقهية وجهات الفتوى الرسمية، ومنها:"),
        "crisis": ("تحدّث مع أحد الآن", "إن كان أحد في خطر فاتصل الآن برقم الطوارئ في بلدك. ويمكنك أيضًا التواصل مع:"),
        "newcomer": ("تحدّث مع أحد", "تواصل مع أقرب مسجد أو مركز إسلامي إليك، فهم يرحبون بك ويساعدونك خطوة خطوة، "
                                    "ويجيبون عن أسئلتك وجهًا لوجه."),
    },
    "en": {
        "fatwa": ("Who to ask", "Ask the official fatwa body in your country, for example:"),
        "finance": ("Who to ask", "Contemporary financial matters are studied by the fiqh academies and official fatwa bodies, for example:"),
        "crisis": ("Talk to someone now", "If anyone is in danger, call your local emergency number now. You can also reach:"),
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
    bodies = {"fatwa": FATWA_BODIES, "finance": [FIQH_ACADEMY, *FATWA_BODIES],
              "crisis": CRISIS_LINES, "newcomer": []}[kind]
    links = [{"label": b[lang], "where": b[f"where_{lang}"], "url": b["url"]} for b in bodies]
    return {"kind": kind, "title": title, "intro": intro, "links": links}

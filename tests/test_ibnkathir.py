"""Ibn Kathir's Tafsir (21 October 2026): passages name their surah and verses, the editor's footnote numbers go,
and Ibn Kathir's own words stay as they are. Synthetic text only."""
from muhawir import ibnkathir

FILE = """#META# 020.BookTITLE	:: تفسير القرآن العظيم
#META#Header#End#
### | مقدمة المحقق
# كلام المحقق لا يُنقل.
### | تفسير سورة الكهف
# وهي مكية (1) .
### || {الحمد لله الذي أنزل على عبده الكتاب ولم يجعل له عوجا (1) قيما لينذر بأسا شديدا (2)}
# قد تقدم في أول التفسير أنه تعالى يحمد نفسه (3) عند فواتح PageV05P135
~~الأمور وخواتيمها.
### || فصل
# في فضل السورة.
### || {أم حسبت أن أصحاب الكهف والرقيم كانوا من آياتنا عجبا (9)}
# هذا إخبار عن قصة أصحاب الكهف.
"""


def test_passages_name_the_surah_and_the_verses_they_explain():
    _, passages = ibnkathir.build(FILE, min_passages=1)
    where = [p["location"].split(" (")[0] for p in passages]
    assert where == ["تفسير سورة الكهف", "تفسير سورة الكهف، الآيات 1–2", "تفسير سورة الكهف، الآيات 1–2، فصل",
                     "تفسير سورة الكهف، الآية 9"]
    assert passages[1]["keywords"] == "سورة الكهف؛ الآيات 1–2" and passages[1]["id"] == "ik:2"


def test_the_editors_preface_and_footnote_numbers_are_left_out():
    _, passages = ibnkathir.build(FILE, min_passages=1)
    text = " ".join(p["text"] for p in passages)
    assert "المحقق" not in text and "(3)" not in text and "PageV" not in text
    assert "يحمد نفسه عند فواتح الأمور وخواتيمها." in text

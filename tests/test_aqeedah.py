"""The creed books from OpenITI mARkdown: markers, page headings and tags removed, wording kept."""
import pytest

from muhawir import aqeedah
from muhawir.corpus import parse_corpus

KHUZAYMA = """######OpenITI#
#META# 020.BookTITLE	:: كتاب التوحيد وإثبات صفات الرب عز وجل
#META#Header#End#

### | باب ذكر إثبات النفس لله
# PageV01P013
# حدثنا يعقوب، قال: ثنا أبو معاوية، عن أبي هريرة، قال: قال رسول الله صلى الله عليه وسلم
### | [ص: 16]
# ، قال: «أنا مع عبدي حين يذكرني» PageV01P016
"""

TAHAWI = """######OpenITI#
#META# 020.BookTITLE	:: متن العقيدة الطحاوية
#META#Header#End#

# $ بسم الله الرحمن الرحيم <span class="matn">هذا ذكر بيان عقيدة أهل السنة</span>
# 103 ودين الله واحد وهو دين الإسلام قال الله تعالى @QB@ إن الدين عند الله الإسلام @QE@ PageV01P060
"""


def test_a_page_heading_does_not_split_a_report_and_the_wording_is_kept():
    _, ps = aqeedah.build("tawhid-ibn-khuzayma", KHUZAYMA, min_passages=1)
    assert len(ps) == 1 and ps[0]["kind"] == "aqeedah" and ps[0]["id"] == "uk:1"
    assert ps[0]["location"] == "باب ذكر إثبات النفس لله (ص16)"
    assert "[ص:" not in ps[0]["text"] and "«أنا مع عبدي حين يذكرني»" in ps[0]["text"]


def test_tags_and_dollar_are_removed_and_verse_marks_become_quran_brackets():
    _, ps = aqeedah.build("aqeeda-tahawiyya", TAHAWI, min_passages=1)
    text = ps[0]["text"]
    assert "<span" not in text and "$" not in text and "@Q" not in text
    assert "﴿ إن الدين عند الله الإسلام ﴾" in text
    assert ps[0]["location"] == "ص60"


def test_sources_name_author_edition_and_licence():
    source, ps = aqeedah.build("tawhid-ibn-khuzayma", KHUZAYMA, min_passages=1)
    about = parse_corpus({"synthetic": False, "sources": [source], "passages": ps}).sources[source["id"]].about
    assert "ابن خزيمة" in about or "بن خزيمة" in about
    assert "مكتبة الرشد" in about and "OpenITI" in about and "CC BY-NC-SA 4.0" in about
    for book in aqeedah.BOOKS.values():
        assert "(ت " in book["about"] and "هـ" in book["about"]


def test_wrong_or_short_file_is_refused():
    with pytest.raises(aqeedah.ImportError_):
        aqeedah.build("usul-al-sunna", KHUZAYMA, min_passages=1)  # another book's file
    with pytest.raises(aqeedah.ImportError_):
        aqeedah.build("tawhid-ibn-khuzayma", KHUZAYMA)  # far fewer passages than the book

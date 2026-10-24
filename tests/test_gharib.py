"""al-Raghib's «المفردات» (21 October 2026): one passage per root, the lines of an entry joined, and the headings
the digital text made of words cut in the middle of an entry put back into it. Synthetic text only."""
from muhawir import gharib

FILE = """#META# 020.BookTITLE	:: المفردات في غريب القرآن
#META#Header#End#
### | مقدمة المحقق
# هذا كلام المحقق ولا يُنقل.
### | كتاب الراء
### | رشد
# الرشد: خلاف الغي، يستعمل استعمال الهداية «1» ، قال: لعلهم يرشدون
~~[البقرة/ 186] . PageV01P354
### | رهط
# الرهط: العصابة دون العش
### | رة، و
# قيل: إلى الأربعين، قال: تسعة رهط [النمل/ 48] .
### | 12-
# بيت من الشعر
~~يتم به الكلام.
### | ركب
# الركوب في الأصل: كون الإنسان على ظهر حيوان، وقد يستعمل في السفينة، وقيل: غير ذلك. والعشرة عدد، ومنه العشرة.
### | مقدمة للفهارس الفنية
# فهارس لا تُنقل.
"""


def test_each_root_is_one_entry_and_the_editors_pages_are_left_out():
    entries = gharib.parse(FILE)
    assert [e["root"] for e in entries] == ["رشد", "رهط", "ركب"]
    assert entries[0]["text"].startswith("الرشد: خلاف الغي") and "«1»" not in entries[0]["text"]
    assert "PageV" not in entries[0]["text"] and "المحقق" not in " ".join(e["text"] for e in entries)
    assert "فهارس" not in entries[-1]["text"]


def test_a_heading_cut_from_the_middle_of_an_entry_goes_back_into_it():
    rahṭ = gharib.parse(FILE)[1]["text"]
    assert "العصابة دون العشرة، وقيل: إلى الأربعين" in rahṭ
    assert "بيت من الشعر يتم به الكلام" in rahṭ  # «12-» numbers a line of verse: not a root


def test_passages_name_the_root_and_carry_their_kind():
    _, passages = gharib.build(FILE, min_entries=1)
    assert passages[0]["location"].startswith("مادة «رشد»") and passages[0]["kind"] == "gharib"
    assert passages[0]["keywords"] == "رشد" and passages[0]["id"] == "g:1"

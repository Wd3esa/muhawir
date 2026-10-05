"""Ibn Hisham's Sira from OpenITI mARkdown: markers removed, wording and pages kept, poem sections left out."""
import pytest

from muhawir import sira
from muhawir.corpus import parse_corpus

SAMPLE = """######OpenITI#
#META# 020.BookTITLE	:: السيرة النبوية لابن هشام
#META#Header#End#

### | غزوة بدر الكبرى
### || (خروج الرسول) :
# قال ابن إسحاق: ثم خرج رسول الله صلى الله عليه وسلم في ليال
~~مضت من شهر رمضان ms0201 في أصحابه. PageV01P606
# وكان معهم سبعون بعيرا. PageV01P607
### || (شعر حسان في يوم بدر) :
# بيت من الشعر لا يُستورد
~~وبيت آخر. PageV01P608
### | ما قيل من الشعر يوم بدر
### || (قصيدة أخرى) :
# شعر لا يُستورد أيضا. PageV01P609
### | غزوة أحد
# قال ابن هشام: وكانت أحد يوم السبت. PageV02P60
"""


def test_markers_are_removed_and_wording_kept():
    _, ps = sira.build(SAMPLE, min_passages=1)
    assert ps[0]["text"] == ("قال ابن إسحاق: ثم خرج رسول الله صلى الله عليه وسلم في ليال "
                             "مضت من شهر رمضان في أصحابه.\nوكان معهم سبعون بعيرا.")
    assert all("ms0" not in p["text"] and "PageV" not in p["text"] and "~~" not in p["text"] for p in ps)


def test_location_names_chapter_heading_volume_and_pages():
    _, ps = sira.build(SAMPLE, min_passages=1)
    assert ps[0]["location"] == "غزوة بدر الكبرى، خروج الرسول (ج1، ص606–607)"
    assert ps[0]["keywords"] == "غزوة بدر الكبرى؛ خروج الرسول"
    assert ps[0]["id"] == "s:1" and ps[0]["kind"] == "seerah"
    assert ps[-1]["location"] == "غزوة أحد (ج2، ص60)"


def test_poem_sections_are_left_out_but_the_narrative_after_them_is_kept():
    _, ps = sira.build(SAMPLE, min_passages=1)
    assert [p["id"] for p in ps] == ["s:1", "s:2"]
    assert not any("لا يُستورد" in p["text"] for p in ps)
    assert sira.is_poem("شعر حسان في يوم بدر") and sira.is_poem("قصيدة أخرى") and sira.is_poem("ما قيل من الشعر يوم بدر")
    assert not sira.is_poem("خروج الرسول")


def test_passages_join_into_a_valid_corpus_with_attribution():
    source, ps = sira.build(SAMPLE, min_passages=1)
    parsed = parse_corpus({"synthetic": False, "sources": [source], "passages": ps})
    about = parsed.sources[sira.SOURCE_ID].about
    assert "بن هشام" in about and "بن إسحاق" in about and "OpenITI" in about and "CC BY-NC-SA 4.0" in about
    assert "متفاوتة في الصحة" in about


def test_truncated_or_wrong_file_is_refused():
    with pytest.raises(sira.ImportError_):
        sira.build(SAMPLE)  # far fewer passages than the real book
    with pytest.raises(sira.ImportError_):
        sira.build(SAMPLE.replace("السيرة النبوية لابن هشام", "كتاب آخر"), min_passages=1)

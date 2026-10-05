"""«بينات» from its PDF: lines rebuilt from glyphs (ligatures, lost alefs, honorifics, verses, numbers),
and each question split into its parts with its number, title, page and keywords."""
import pytest

from muhawir import bayyinat
from muhawir.corpus import parse_corpus


class FakePage:
    """A page whose text trace is given glyph by glyph, left to right as PyMuPDF reports it."""

    def __init__(self, spans):
        self.spans = spans

    def get_texttrace(self):
        return self.spans


def span(font, glyphs, y=100.0, size=15.0):
    """glyphs: (text, glyph id, x0, x1) in visual order; a glyph id of -1 continues a ligature."""
    return {"font": font, "size": size,
            "chars": [(ord(t), gid, (x0, y), (x0, y - 10, x1, y)) for t, gid, x0, x1 in glyphs]}


def test_a_line_is_read_right_to_left_with_ligatures_marks_and_numbers():
    # «الله أكبر 40» drawn right to left: the ligature «لله» is one glyph followed by its letters
    glyphs = [("4", 9, 10, 14), ("0", 9, 14, 18), (" ", 3, 18, 21),
              ("ر", 1, 21, 26), ("ب", 2, 26, 29), ("ك", 3, 29, 34), ("َ", 5, 34, 34), ("أ", 4, 34, 37), (" ", 3, 37, 40),
              ("ل", 257, 40, 52), ("ل", -1, 40, 40), ("ه", -1, 40, 40), ("ا", 208, 52, 55)]
    lines = bayyinat._page_lines(FakePage([span("adwa-assalaf", glyphs)]))
    assert [text for _y, _f, _s, text in lines] == ["الله أكبر 40"]  # the fatha is left out


def test_a_bold_lam_alef_gets_its_alef_back_and_honorifics_are_written_out():
    glyphs = [("ل", 239, 10, 18), ("و", 1, 18, 24), ("أ", 2, 24, 28)]  # «أول» + lost alef = «أولا»
    honor = [("h", 73, 0, 8)]
    lines = bayyinat._page_lines(FakePage([span("adwaassalaf-Bold", glyphs),
                                           span("KFGQPCArabicSymbols01", honor)]))
    assert lines[0][3].split() == ["أولا", "رضي", "الله", "عنه"]


def test_a_verse_in_the_quran_font_becomes_a_marked_gap_never_text():
    glyphs = [("", 1, 30, 60), ("", 2, 60, 90)]
    lines = bayyinat._page_lines(FakePage([span("adwa-assalaf", [("﴾", 5, 20, 30)]), span("QCF4_Hafs_02", glyphs),
                                           span("adwa-assalaf", [("﴿", 5, 90, 100)])]))
    assert bayyinat._clean(lines[0][3] + " [البقرة: 2]") == "(الآية) [البقرة: 2]"


def L(text, font="adwa-assalaf", size=15, page=21, y=100):
    return {"page": page, "y": y, "font": font, "size": size, "text": text}


SAMPLE = [
    L("الم)1لة ل1س:  إنكار وجود الله تعالى.", "AbdoLine", 16, y=80),
    L("السؤال", "DINNextLTW23-Medium", 14, y=120),
    L("لماذا نؤمن بوجود إله؟", y=150),
    L("عبارات مشاإهة لل)ؤال", "(AH)-Manal-High", 14, y=190),
    L("هل هناك حاجة لرب مدبر؟", y=220),
    L("الجواب", "DINNextLTW23-Medium", 14, y=260),
    L("مختصرم ا جاإة:", "(AH)-Manal-High", 14, y=300),
    L("الأدلة على وجود الله كثيرة.", y=330),
    L("الجوابم التفصيلي:", "(AH)-Manal-High", 14, y=380),
    L("فخلق السموات والأرض:  ", y=410),
    L("[يس: 40]، يدل على خالق.", y=432),
    L("وهذا آخر الكلام.", page=22, y=80),
    L("كلمات دلالية: وجود الله، الإيمان بالله.", size=13, page=22, y=120),
]


def test_a_question_is_split_into_its_parts_with_number_title_pages_and_keywords():
    _, ps = bayyinat.build(SAMPLE, min_questions=1)
    assert [p["location"] for p in ps] == [
        "المسألة (1): إنكار وجود الله تعالى. — السؤال (ص21)",
        "المسألة (1): إنكار وجود الله تعالى. — مختصر الإجابة (ص21)",
        "المسألة (1): إنكار وجود الله تعالى. — الجواب التفصيلي (ص21–22)"]
    assert ps[0]["text"] == "السؤال: لماذا نؤمن بوجود إله؟"
    assert ps[2]["text"] == "فخلق السموات والأرض: (الآية) [يس: 40]، يدل على خالق.\nوهذا آخر الكلام."
    assert "هل هناك حاجة لرب مدبر؟" in ps[0]["keywords"] and "الإيمان بالله" in ps[0]["keywords"]
    assert all(p["kind"] == "qa" and p["id"].startswith("b:") for p in ps)


def test_source_names_the_book_and_its_rights_and_a_short_file_is_refused():
    source, ps = bayyinat.build(SAMPLE, min_questions=1)
    about = parse_corpus({"synthetic": False, "sources": [source], "passages": ps}).sources["bayyinat"].about
    assert "مركز أصول" in about and "حقوقه لأصحابها" in about and "(الآية)" in about
    with pytest.raises(bayyinat.ImportError_):
        bayyinat.build(SAMPLE)


def test_a_wrong_pdf_is_refused(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7 not the book")
    with pytest.raises(bayyinat.ImportError_):
        bayyinat.read_pdf(pdf)

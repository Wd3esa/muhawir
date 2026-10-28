"""Fatwa collections from OpenITI mARkdown, and contemporary financial matters: the published fatwas are
quoted with their authors, Muhawir gives no ruling of its own on the product asked about, and refers on."""
import json

import pytest

from muhawir import classify, fatawa
from muhawir.corpus import parse_corpus
from muhawir.generate import ModelGenerator
from muhawir.pipeline import REFERRED, Muhawir

SAMPLE = """######OpenITI#
#META# 020.BookTITLE	:: مجموع فتاوى العلامة عبد العزيز بن باز
#META#Header#End#

### | حكم بيع التقسيط
# س: ما حكم البيع بالتقسيط مع زيادة الثمن؟
~~ج: لا حرج في ذلك إذا كان الأجل معلوما. PageV19P15
### | 1 -
# أن تكون السلعة مملوكة للبائع. PageV19P16
### | حكم آخر
# س: سؤال آخر. PageV19P17
"""


def test_markers_are_removed_wording_kept_and_list_items_stay_in_their_fatwa():
    _, ps = fatawa.build("fatawa-ibn-baz", SAMPLE, min_passages=1)
    assert ps[0]["text"] == ("س: ما حكم البيع بالتقسيط مع زيادة الثمن؟ ج: لا حرج في ذلك إذا كان الأجل معلوما.\n"
                             "1 -\nأن تكون السلعة مملوكة للبائع.")
    assert ps[0]["location"] == "حكم بيع التقسيط (ج19، ص15–16)"
    assert ps[0]["id"] == "z:1" and ps[0]["kind"] == "fatwa" and ps[0]["keywords"] == "حكم بيع التقسيط"
    assert ps[1]["location"] == "حكم آخر (ج19، ص17)"
    assert all("PageV" not in p["text"] and "~~" not in p["text"] for p in ps)


def test_source_record_names_the_author_and_discloses_the_rights():
    source, ps = fatawa.build("fatawa-ibn-baz", SAMPLE, min_passages=1)
    about = parse_corpus({"synthetic": False, "sources": [source], "passages": ps}).sources["fatawa-ibn-baz"].about
    assert "بن باز" in about and "OpenITI" in about and "حقوق نصه لأصحابها" in about


def test_wrong_or_truncated_file_is_refused():
    with pytest.raises(fatawa.ImportError_):
        fatawa.build("fatawa-ibn-baz", SAMPLE)  # far fewer passages than the real collection
    with pytest.raises(fatawa.ImportError_):
        fatawa.build("fatawa-ibn-uthaymeen", SAMPLE, min_passages=1)  # another book's file


def test_named_buy_now_pay_later_companies_and_instalment_rulings_are_contemporary():
    # read by the understanding step from the meaning (28 October 2026), not by a list of words
    for q in ("هل تابي وتمارا ربوية بناءً على المصادر المتاحة", "ما حكم الشراء بالتقسيط؟", "Is Tabby halal?"):
        assert classify.check(q).kind is None, q
    assert classify.names_company("تابي شركة") and not classify.names_company("بيع التقسيط")


# --- the reply: quoted fatwas, no ruling of Muhawir's own -------------------------------------------

FATWA = "س: ما حكم البيع بالتقسيط مع زيادة الثمن؟ ج: لا حرج في ذلك إذا كان الأجل معلوما."
CORPUS = parse_corpus({"synthetic": True, "sources": [
    {"id": "fatawa-ibn-baz", "name": "مجموع فتاوى ابن باز", "about": "مصدر تجريبي.", "url": ""}],
    "passages": [{"id": "z:1", "source_id": "fatawa-ibn-baz", "kind": "fatwa", "location": "حكم بيع التقسيط (ج19، ص15)",
                  "text": FATWA, "keywords": "حكم بيع التقسيط"}]})


def _engine(claims):
    calls = []

    def call(system, user, schema=None):
        calls.append((system, user))
        if '"question"' in json.dumps(schema or {}):  # the understanding step reads a contemporary financial matter
            asked = user.rsplit("<<<", 1)[-1].split(">>>")[0]
            return json.dumps({"question": asked, "kind": "contemporary", "queries": ["بيع التقسيط"]}, ensure_ascii=False)
        if "queries" in json.dumps(schema or {}):
            return '{"queries": ["بيع التقسيط"]}'
        return json.dumps({"abstain": False, "claims": claims}, ensure_ascii=False)
    return Muhawir(CORPUS, ModelGenerator([("m", call)])), calls


def test_instalment_question_quotes_the_fatwa_and_refers_with_the_finance_card():
    sentence = "سُئل الشيخ ابن باز عن البيع بالتقسيط مع زيادة الثمن فأجاب بأنه لا حرج في ذلك إذا كان الأجل معلوما."
    engine, calls = _engine([{"text": sentence, "passage_ids": ["z:1"]}])
    res = engine.ask("ما حكم الشراء بالتقسيط؟")
    assert res.status == REFERRED and res.message.startswith("لا أحكم بنفسي")
    assert [c["text"] for c in res.claims] == [sentence]
    assert res.referral["kind"] == "finance"
    assert any("مسألة مالية معاصرة" in user for _system, user in calls)  # the writing step is told how to quote


def test_a_sentence_applying_the_fatwa_to_the_company_is_dropped():
    engine, _ = _engine([{"text": "إذن تابي لا حرج فيها لأن الأجل معلوم.", "passage_ids": ["z:1"]}])
    res = engine.ask("هل تابي وتمارا ربوية بناءً على المصادر المتاحة")
    assert res.status == REFERRED and res.claims == []
    assert res.referral["kind"] == "finance"


def test_a_fatwa_on_another_product_is_not_quoted_under_the_question():
    # live site, 6 October 2026: «حكم التداول بالعملات الرقمية» quoted fatwas on paper money and bank interest
    sentence = "سُئل الشيخ ابن باز عن البيع بالتقسيط مع زيادة الثمن فأجاب بأنه لا حرج في ذلك إذا كان الأجل معلوما."
    engine, _ = _engine([{"text": sentence, "passage_ids": ["z:1"]}])
    res = engine.ask("ما حكم التداول بالعملات الرقمية؟")
    assert res.status == REFERRED and res.claims == [] and not res.sources
    assert res.referral["kind"] == "finance"


def test_a_paragraph_cut_in_the_middle_of_a_sentence_goes_on_in_the_same_passage():
    text = SAMPLE.replace("### | حكم آخر\n# س: سؤال آخر. PageV19P17\n",
                          "### | أركان الإسلام\n# وقد بين النبي PageV23P326\n"
                          "# صلى الله عليه وسلم أنها خمسة. PageV23P327\n# س: سؤال آخر. PageV23P327\n")
    _, ps = fatawa.build("fatawa-ibn-baz", text, min_passages=1)
    arkan = [p for p in ps if p["keywords"] == "أركان الإسلام"][0]["text"]
    assert arkan == "وقد بين النبي صلى الله عليه وسلم أنها خمسة.\nس: سؤال آخر."


def test_words_made_into_a_heading_in_the_middle_of_a_question_stay_with_their_question():
    text = SAMPLE.replace("### | حكم آخر\n# س: سؤال آخر. PageV19P17\n",
                          "### | حكم آخر\n# ج: لا بأس بذلك. س11: ما هي\n### | العبارات التي تطلق في حق الأموات\n"
                          "# ، فنحن نسمع (المرحوم) فهل هي صحيحة؟ PageV19P17\n")
    _, ps = fatawa.build("fatawa-ibn-baz", text, min_passages=1)
    assert ps[1]["text"] == "ج: لا بأس بذلك."
    assert ps[2]["text"] == "س11: ما هي العبارات التي تطلق في حق الأموات، فنحن نسمع (المرحوم) فهل هي صحيحة؟"
    assert ps[2]["keywords"] == "العبارات التي تطلق في حق الأموات"


def test_a_heading_cut_in_the_middle_opens_its_text_whole():
    text = SAMPLE.replace("### | حكم آخر\n# س: سؤال آخر. PageV19P17\n",
                          "### | مشروعية الصلاة على النبي\n# صلى الله عليه وسلم إذا مر ذكره في الخطبة PageV19P17\n")
    _, ps = fatawa.build("fatawa-ibn-baz", text, min_passages=1)
    assert ps[1]["text"] == "مشروعية الصلاة على النبي صلى الله عليه وسلم إذا مر ذكره في الخطبة"


def test_the_creed_books_keep_their_paragraphs_as_they_are():
    """The old creed texts have few sentence marks: joining «unfinished» paragraphs made «العقيدة الطحاوية» five
    passages instead of eleven and stopped the build (21 October 2026). They are parsed as before."""
    from muhawir.fatawa import parse
    text = "كتاب #META#Header#End#\n### | باب\n# الأول بلا علامة\n# الثاني بلا علامة\n# الثالث بلا علامة\n"
    assert len(parse(text, "كتاب", join_cut=False)) == 3

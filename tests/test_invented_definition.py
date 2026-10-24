"""A definition of what was asked must be in the cited passages' own words (21 October 2026: «أركان الإسلام هي
الأساسات التي بُني عليها الإسلام» passed the second reading). Synthetic data only."""
import pytest

from muhawir import wordforms
from muhawir.verify import invented_definition

ARKAN = ["قال الشيخ: أركان الإسلام خمسة وهي أركانه الظاهرة كما قال النبي صلى الله عليه وسلم: بني الإسلام على خمس"]
ZAKAT = ["الزكاة حق واجب في المال يصرف للفقراء والمساكين إذا بلغ النصاب وحال عليه الحول"]


def test_a_definition_in_words_the_source_does_not_use_is_caught():
    assert invented_definition("أركان الإسلام هي العادات الاجتماعية التي يتوارثها الناس.", ARKAN, "اركان الاسلام")
    assert invented_definition("الزكاة هي أن تساعد الناس المحتاجين بمالك.", ZAKAT, "ما هي الزكاة؟")


@pytest.fixture(autouse=True)
def few_synonyms(tmp_path, monkeypatch):
    """A small synthetic WordNet in the real file's format, so the tests do not depend on the downloaded one."""
    file = tmp_path / "wn.tab"
    file.write_text("# test\n1-n\tarb:lemma\tأساس\n1-n\tarb:lemma\tرُكن\n1-n\tarb:lemma\tقاعدة\n", encoding="utf-8")
    monkeypatch.setattr(wordforms, "SYNONYMS_FILE", file)
    wordforms._wordnet.cache_clear()
    yield
    wordforms._wordnet.cache_clear()


def test_a_definition_in_the_sources_words_passes_in_any_form_or_synonym_of_them():
    assert invented_definition("أركان الإسلام هي الأساسات التي بُني عليها الإسلام.", ARKAN, "اركان الاسلام") == []
    assert invented_definition("الزكاة هي مال يُعطى للمحتاجين.", ZAKAT + ["وتعطى لمن احتاج إليها"], "ما الزكاة") == []
    assert invented_definition("أركان الإسلام هي أركانه الظاهرة الخمسة التي بني عليها.", ARKAN, "اركان الاسلام") == []
    assert invented_definition("الزكاة هي حق واجب في المال يُعطى للفقراء.", ZAKAT, "ما هي الزكاة؟") == []
    assert invented_definition("يُقصد بالزكاة حق واجب في مال المسلم.", ZAKAT, "ما معنى الزكاة") == []


def test_a_gloss_of_another_word_and_a_sentence_that_defines_nothing_are_left_alone():
    assert invented_definition("الحول هو سنة كاملة.", ZAKAT, "ما شروط الزكاة؟") == []  # not what was asked
    assert invented_definition("يخبرنا الله تعالى أنه هو الأول.", ["هو الأول والآخر"], "من هو الله") == []
    assert invented_definition("وتجب الزكاة إذا بلغ المال النصاب.", ZAKAT, "ما هي الزكاة؟") == []


def test_words_the_code_cannot_match_are_dropped_only_when_the_model_confirms(monkeypatch):
    import json
    from muhawir.corpus import parse_corpus
    from muhawir.generate import ModelGenerator
    from muhawir.pipeline import Muhawir
    from muhawir.verify import Claim
    corpus = parse_corpus({"synthetic": True, "sources": [{"id": "s", "name": "م", "about": "م"}], "passages": [
        {"id": "z:1", "source_id": "s", "location": "أ", "kind": "fatwa", "text": ARKAN[0]}]})
    asked = []

    def model(absent):
        def call(system, user, schema=None):
            if "absent" in json.dumps(schema or {}):
                asked.append(user)
                return json.dumps({"absent": absent}, ensure_ascii=False)
            return json.dumps({"supported": [True]})
        return Muhawir(corpus, ModelGenerator([("m", call)]))

    claim = Claim("أركان الإسلام هي العادات الاجتماعية التي يتوارثها الناس.", ("z:1",))
    kept, rejected = model(["العادات"])._checked([claim], corpus, {"z:1"}, [], "اركان الاسلام")
    assert not kept and "definition not in" in rejected[-1].reason and asked
    kept, _ = model([])._checked([claim], corpus, {"z:1"}, [], "اركان الاسلام")
    assert kept == [claim]  # the model found every word in the passage's meaning: the sentence stays


def test_a_hadith_is_credited_only_to_a_book_the_sentence_cites():
    from muhawir.verify import wrong_collector
    both = {"sahih-bukhari", "sahih-muslim"}
    assert wrong_collector("روى البخاري ومسلم أن النبي ﷺ قال: بني الإسلام على خمس", both) == []
    assert wrong_collector("روى البخاري ومسلم أن النبي ﷺ قال كذا", {"sahih-muslim"}) == ["البخاري"]
    assert wrong_collector("وفي صحيح مسلم أن النبي ﷺ قال كذا", {"fatawa-ibn-baz"}) == ["مسلم"]
    assert wrong_collector("قال النبي ﷺ كذا", {"fatawa-ibn-baz"}) == []

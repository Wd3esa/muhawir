"""Display boundary tests: no downloads or real model calls."""
from types import SimpleNamespace
import pytest
from muhawir import tashkeel as t

@pytest.fixture(autouse=True)
def reset_limits():
    t._CALLS.clear()
    t._predict.cache_clear()
    yield
    t._stop()


def test_mixed_content():
    text = 'هذا كتاب. English 12 اردو'
    result, ok = t.merge(text, 'هَذَا كِتَاب. English 12 اردو', ())
    assert ok and t.bare(result) == text
    assert result.endswith('English 12 اردو')

@pytest.mark.parametrize('prediction', ['هذا كتاب جديد.', 'هدا كتاب.', 'هذا.', 'كتاب هذا.'])
def test_rejects_nonmark_changes(prediction):
    assert t.merge('هذا كتاب.', prediction, ()) == ('هذا كتاب.', False)


def test_existing_marks_win():
    assert t.merge('عَلِمَ الرجل', 'عُلِمَ الرَّجُل', ())[0] == 'عَلِمَ الرَّجُل'

@pytest.mark.parametrize('text', ['قال «هذا كتاب»', 'قال ﴿هذا كتاب﴾', 'قال “هذا كتاب”', 'قال "هذا كتاب"', 'قال «هذا كتاب'])
def test_quotes(text):
    assert t.merge(text, text.replace('هذا كتاب', 'هَذَا كِتَاب'), ())[0] == text


def test_source_marks():
    text = 'قال إنما الأعمال بالنيات.'
    result, ok = t.merge(text, 'قَال إِنَّمَا الأَعْمَال بِالنِّيَّات.', ('إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ',))
    assert ok and result == 'قَال إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ.'
    assert t.bare(result) == t.bare(text)


def test_no_incidental_match():
    assert t.protect('هذا كتاب مفيد', ('قَرَأَ كِتَابَ التَّارِيخِ أَمْسِ',)) == {}


def fake_python(tmp_path, body):
    path = tmp_path / 'python'
    path.write_text('#!/bin/bash\n' + body)
    path.chmod(0o755)
    return path


ECHO = 'echo \'{"ready": true}\'\nwhile read -r line; do echo "$line"; done\n'


def test_timeout(monkeypatch, tmp_path):
    monkeypatch.setenv('MUHAWIR_TASHKEEL_PYTHON', str(fake_python(tmp_path, 'echo \'{"ready": true}\'\nsleep 30\n')))
    monkeypatch.setattr(t, '_RUN', 0.5)
    with pytest.raises(t.Unavailable): t.diacritize(['هذا كتاب'])
    assert not t._SLOT.locked() and t._predict.cache_info().currsize == 0 and t._proc is None


def test_worker_is_reused_and_restarted(monkeypatch, tmp_path):
    monkeypatch.setenv('MUHAWIR_TASHKEEL_PYTHON', str(fake_python(tmp_path, ECHO)))
    assert t.diacritize(['هذا كتاب'])['texts'] == ['هذا كتاب']
    first = t._proc
    assert t.diacritize(['هذا كتاب جديد'])['texts'] == ['هذا كتاب جديد']
    assert t._proc is first                      # same process: the model is not loaded again
    first.kill(); first.wait()
    assert t.diacritize(['كتاب آخر'])['texts'] == ['كتاب آخر']
    assert t._proc is not first                  # a dead worker is replaced


def test_worker_that_never_starts(monkeypatch, tmp_path):
    monkeypatch.setenv('MUHAWIR_TASHKEEL_PYTHON', str(fake_python(tmp_path, 'exit 1\n')))
    with pytest.raises(t.Unavailable): t.diacritize(['سلام'])
    assert t._proc is None


def test_missing_environment(monkeypatch, tmp_path):
    monkeypatch.setenv('MUHAWIR_TASHKEEL_PYTHON', str(tmp_path / 'none'))
    with pytest.raises(t.Unavailable): t.diacritize(['سلام'])


def test_idle_stops_worker(monkeypatch, tmp_path):
    monkeypatch.setenv('MUHAWIR_TASHKEEL_PYTHON', str(fake_python(tmp_path, ECHO)))
    monkeypatch.setattr(t, '_IDLE', 0.2)
    t.diacritize(['سلام'])
    import time; time.sleep(0.8)
    assert t._proc is None


def test_rejected_prediction(monkeypatch):
    monkeypatch.setattr(t, '_predict', lambda texts: tuple('كلام مختلف' for _ in texts))
    result = t.diacritize(['هذا كتاب'])
    assert result['texts'] == ['هذا كتاب'] and result['rejected'] == 1


def test_limits(monkeypatch):
    with pytest.raises(ValueError): t.diacritize(['ا' * 6001])
    monkeypatch.setattr(t, '_WAIT', 0.2)
    t._SLOT.acquire()
    try:
        with pytest.raises(t.Unavailable): t.diacritize(['سلام'])   # waits, then gives up
        monkeypatch.setattr(t, '_QUEUE', 0)
        with pytest.raises(t.Unavailable): t.diacritize(['سلام'])   # queue full: turned away at once
    finally: t._SLOT.release()


def test_request_waits_for_its_turn(monkeypatch):
    import threading, time
    monkeypatch.setattr(t, '_predict', lambda texts: texts)
    t._SLOT.acquire()
    threading.Timer(0.3, t._SLOT.release).start()
    start = time.monotonic()
    assert t.diacritize(['سلام'])['texts'] == ['سلام']
    assert time.monotonic() - start >= 0.25


@pytest.fixture
def api(monkeypatch):
    from muhawir import corpus
    from fastapi.testclient import TestClient
    fake = corpus.Corpus({'s': corpus.Source('s', 'Test', 'Synthetic')},
        [corpus.Passage('p', 's', 'test', 'إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ')], True)
    monkeypatch.setattr(corpus, 'load_corpus', lambda path: fake)
    from muhawir import server
    monkeypatch.setattr(server, 'engine', SimpleNamespace(corpus=fake))
    return TestClient(server.app)


def test_api_sources_without_generator(api, monkeypatch):
    monkeypatch.setattr(t, '_predict', lambda texts: texts)
    response = api.post('/api/tashkeel', json={'texts': ['إنما الأعمال بالنيات'], 'passage_ids': ['p']})
    assert response.status_code == 200
    assert response.json()['texts'] == ['إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ']


def test_api_validation(api):
    for body in [{'texts': ['ا' * 6001]}, {'texts': []}, {'texts': [4]}, {'texts':['سلام'], 'passage_ids':['p'*201]}]:
        assert api.post('/api/tashkeel', json=body).status_code == 422


def test_api_unavailable(api, monkeypatch):
    def fail(*args, **kwargs): raise t.Unavailable('offline')
    monkeypatch.setattr(t, 'diacritize', fail)
    assert api.post('/api/tashkeel', json={'texts':['سلام']}).status_code == 503


def test_omitted_final_period_restored():
    text = 'هذا كتاب.'
    output, ok = t.merge(text, 'هَذَا كِتَاب', ())
    assert ok and output == 'هَذَا كِتَاب.'
    assert t.bare(output) == text


def test_omitted_period_cannot_hide_changed_letter():
    assert t.merge('هذا كتاب.', 'هدا كتاب', ()) == ('هذا كتاب.', False)


@pytest.mark.parametrize('prediction', ['هَذَا كِتَاب!', 'هَذَا  كِتَاب', '<b>هَذَا كِتَاب</b>', 'هَذَا كِتَاب'])
def test_punctuation_of_the_tool_is_ignored(prediction):
    # only the vowels of the words are taken; what lies between the words stays as the text had it
    assert t.merge('هذا كتاب.', prediction, ()) == ('هَذَا كِتَاب.', True)


def test_digits_and_citations_stay():
    text = 'الرشد: العقل [1][2]. وقال مالك (2) هو الصلاح.'
    prediction = text.replace('الرشد', 'الرُّشْد').replace('[1][2]', '')
    result, ok = t.merge(text, prediction, ())
    assert ok and '[1][2]' in result and '(2)' in result and result.startswith('الرُّشْد: العقل')
    assert t.bare(result) == text


def test_chunks_join_back():
    text = 'الرشد: العقل [1]. وقال مالك؟\nهو الصلاح! نعم'
    assert ''.join(t.chunks(text)) == text and len(t.chunks(text)) >= 4


def test_one_bad_sentence_does_not_spoil_the_rest(monkeypatch):
    def predict(texts):
        return tuple('كلام مختلف' if 'مالك' in c else c.replace('الرشد', 'الرُّشْد') for c in texts)
    monkeypatch.setattr(t, '_predict', predict)
    result = t.diacritize(['الرشد العقل. قال مالك كذا. الرشد صلاح.'])
    assert result['texts'] == ['الرُّشْد العقل. قال مالك كذا. الرُّشْد صلاح.']
    assert result['rejected'] == 1


def test_long_answer_is_sent_by_sentences(monkeypatch):
    seen = []
    monkeypatch.setattr(t, '_predict', lambda texts: seen.append(texts) or texts)
    t.diacritize(['جملة أولى. جملة ثانية. جملة ثالثة.', 'جملة أولى. أخرى.'])
    assert all(len(c) < 30 for c in seen[0]) and len(seen[0]) == len(set(seen[0]))


def test_unvowelled_source_does_not_cancel_the_tool():
    text = 'ورجح الطبري أن الرشد هو العقل وإصلاح المال.'
    prediction = 'وَرَجَّحَ الطَّبَرِيُّ أَنَّ الرُّشْدَ هُوَ الْعَقْلُ وَإِصْلَاحُ الْمَالِ.'
    result, ok = t.merge(text, prediction, ('قال الطبري إن الرشد هو العقل وإصلاح المال في هذه الآية',))
    assert ok and result == prediction


def test_vowelled_source_still_wins_over_the_tool():
    text = 'قال إنما الأعمال بالنيات.'
    source = 'إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ وَإِنَّمَا لِكُلِّ امْرِئٍ'
    result, ok = t.merge(text, 'قَال إِنَّمَا الأَعْمَال بِالنِّيَّات.', (source,))
    assert ok and 'الْأَعْمَالُ بِالنِّيَّاتِ' in result


def test_a_lone_heading_word_takes_the_mushaf_form_or_stays_bare(monkeypatch):
    """A heading is one word with no sentence: the tool's guess is the worst there, so the mushaf's own form is used,
    and a word the mushaf does not hold keeps no vowels rather than a wrong one. Sentences still go to the tool."""
    asked = []

    def predict(texts):
        asked.extend(texts)
        return tuple('هَذَا كِتَاب.' for _ in texts)

    monkeypatch.setattr(t, '_predict', predict)
    forms = {'غورا': 'غَوْرًا'}
    out = t.diacritize(['غورا', 'بخع', 'هذا كتاب.'], (), forms.get)
    assert out['texts'] == ['غَوْرا', 'بخع', 'هَذَا كِتَاب.']  # a heading has no case ending
    assert asked == ['هذا كتاب.'] and out['rejected'] == 0
    assert t.diacritize(['غورا'], (), lambda w: 'غُوْرًى')['texts'] == ['غورا']  # a form with other letters is refused


def test_a_lone_word_not_in_the_mushaf_takes_its_vowels_from_the_answers_own_sentences(monkeypatch):
    """Any heading word, not a list of words: it is vowelled as the tool vowelled it with a sentence around it, minus
    the article and the case ending; when it is in no sentence it has no vowels."""
    vowelled = {'البخع:': 'الْبَخْعُ:', ' قتل النفس.': ' قَتْلُ النَّفْسِ.'}
    monkeypatch.setattr(t, '_predict', lambda texts: tuple(vowelled[x] for x in texts))
    out = t.diacritize(['بخع', 'قطار', 'البخع: قتل النفس.'], (), lambda word: None)
    assert out['texts'] == ['بَخْع', 'قطار', 'الْبَخْعُ: قَتْلُ النَّفْسِ.']
    assert t._heading_form('شمس', ['الشَّمْسُ طَالِعَةٌ']) == 'شَمْس'
    assert t._heading_form('غورا', ['مَاؤُهَا غَوْرًا']) == 'غَوْرا'


def test_a_heading_from_the_mushaf_loses_the_case_ending_the_verse_gave_it():
    assert t_without('الْكَهْفِ') == 'الْكَهْف'
    assert t_without('مُلْتَحَدًا') == 'مُلْتَحَدا'
    assert t_without('سَمَا') == 'سَمَا'


def t_without(word):
    return t._without_case_ending(word)


def test_ending_is_split_from_the_rest_of_the_word():
    assert t._split_ending('غَوْرًا') == ('غَوْر', 'ً', 'ا')
    assert t._split_ending('الْكَهْفِ') == ('الْكَهْف', 'ِ', '')
    assert t._split_ending('سَمَا') == ('سَمَا', '', '')
    assert t._split_ending('مُلْتَحَدًا') == ('مُلْتَحَد', 'ً', 'ا')


def test_lexicon_keeps_only_what_the_books_agree_on_and_corrects_the_stem_not_the_ending():
    books = ['فِي غَوْرًا', 'مَاؤُهَا غَوْرًا', 'وَأَصْبَحَ غَوْرًا', 'مُلْتَحَدًا', 'مُلْتَحَدًا', 'مُلْتَحَدًا',
             'عِلْمٌ', 'عَلِمَ', 'عَلَّمَ', 'قَوْلُهُ']
    lexicon = t.build_lexicon(books)
    assert lexicon['غورا'] == ('غَوْر', 'ا', 'ً')
    assert 'علم' not in lexicon  # the books do not agree on it: no correction
    assert 'قوله' not in lexicon  # seen once: too few places to trust
    assert t._correct_stem('مُلْتَحِدَا', lexicon) == 'مُلْتَحَدًا'  # stem put right; the tool gave no ending, the books' is used
    assert t._correct_stem('مُلْتَحِدًا', lexicon) == 'مُلْتَحَدًا'
    assert t._correct_stem('غُورَا', lexicon) == 'غَوْرًا'
    assert t._correct_stem('مُلْتَحِدٌ', t.build_lexicon(['مُلْتَحَدٌ'] * 3)) == 'مُلْتَحَدٌ'  # the tool's own ending is kept
    assert t._correct_stem('عَلِمَ', lexicon) == 'عَلِمَ'
    assert t._correct_stem('كلمة', lexicon) == 'كلمة'


def test_merge_corrects_with_a_lexicon_but_never_the_protected_words():
    lexicon = t.build_lexicon(['مُلْتَحَدًا'] * 3)
    text = 'ولن تجد ملتحدا'
    assert t.merge(text, 'وَلَنْ تَجِدَ مُلْتَحِدًا', (), lexicon)[0] == 'وَلَنْ تَجِدَ مُلْتَحَدًا'
    assert t.merge(text, 'وَلَنْ تَجِدَ مُلْتَحِدًا', ())[0] == 'وَلَنْ تَجِدَ مُلْتَحِدًا'
    assert t.merge('قال «ملتحدا»', 'قَالَ «مُلْتَحِدًا»', (), lexicon)[0] == 'قَالَ «ملتحدا»'  # a quoted word is left as written


def test_context_goes_to_the_tool_with_the_sentence_and_is_cut_off_again(monkeypatch):
    sent = []

    def predict(texts):
        sent.extend(texts)
        return tuple(t.WORDS.sub(lambda m: m.group() + 'َ', x) for x in texts)  # a fatha on every word

    monkeypatch.setattr(t, '_predict', predict)
    out = t.diacritize(['معنى قوله: هذا كتاب.'], (), None, context=True)
    assert sent == ['معنى قوله:', 'معنى قوله  هذا كتاب.']  # the second sentence is read with what leads into it
    assert out['texts'] == ['معنىَ قولهَ: هذاَ كتابَ.'] and out['rejected'] == 0
    assert t._left_context(['معنى قوله:']) == 'معنى قوله'
    sent.clear()
    t._predict.cache_clear() if hasattr(t._predict, 'cache_clear') else None
    t.diacritize(['معنى قوله: هذا كتاب.'], (), None, context=False)
    assert sent == ['معنى قوله:', ' هذا كتاب.']


def test_a_word_the_books_never_give_an_ending_gets_none_from_the_tool():
    lexicon = t.build_lexicon(['أَوْ يَقُولُ', 'أَوْ لَا', 'أَوْ هُوَ', 'فِي الْأَرْضِ', 'فِي الدَّارِ', 'فِي كُلِّ'])
    assert t._correct_stem('أَوَ', lexicon) == 'أَوْ'
    assert t._correct_stem('فِيَ', lexicon) == 'فِي'


def test_forbidding_unmarked_endings_leaves_long_vowels_and_the_middle_of_words_alone():
    np = pytest.importorskip('numpy')
    from muhawir import tashkeel_worker as w
    letters = ['<PAD>', '<BOS>', '<EOS>', ' ', 'q', 'w', 'l', 'h', 'A', 'y', '<MASK>']
    ids = np.array([[4, 5, 6, 7, 3, 4, 8]])  # «qwlh qA»: the end of «qwlh» is «h»; the end of «qA» is a bare alef
    logits = np.zeros((1, 7, 4))
    out = w.forbid_unmarked_endings(logits, ids, letters, 0)
    assert out[0, 3, 0] == -np.inf  # «h» ends a word: «no mark» is impossible
    assert out[0, 6, 0] == 0  # the alef may end bare
    assert not np.isinf(out[0, :3, 0]).any() and out[0, 5, 0] == 0  # middle of a word, and the first of «qA»
    assert logits[0, 3, 0] == 0  # the input is not changed
    out = w.forbid_unmarked_endings(logits, np.array([[4, 7]]), letters, 0)  # the last letter of the text counts as an end
    assert out[0, 1, 0] == -np.inf


def test_mostly_vowelled_tells_a_vowelled_book_from_a_few_stray_marks():
    assert t.mostly_vowelled("قَالَ رَسُولُ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ")
    assert not t.mostly_vowelled("قال رسول الله صلى الله عليه وسلم فِي الْمَسْأَلَةِ")
    assert not t.mostly_vowelled("")


def test_pairs_decide_a_word_the_single_word_cannot():
    # «يحل» is read two ways on its own; after «أن» and after «لا» the books read it differently
    books = ["أَنْ يُحِلَّ اللَّهُ"] * 3 + ["لَا يَحِلُّ لَكُمْ"] * 3
    pairs = t.build_pairs(books)
    assert pairs[("أن", "يحل")][0] == "يُحِلَّ"
    assert pairs[("لا", "يحل")][0] == "يَحِلُّ"
    assert t.build_lexicon(books).get("يحل") is None  # the word alone is ambiguous: the lexicon says nothing


def test_merge_uses_the_pair_before_the_single_word():
    pairs = {("أن", "يحل"): ("يُحِلَّ", "", "")}
    out, ok = t.merge("أن يحل", "أَنْ يَحُلُّ", (), None, pairs)
    assert ok and out.split()[1] == "يُحِلَّ"
    out, ok = t.merge("لا يحل", "لَا يَحُلُّ", (), None, pairs)  # another word before: the tool's reading stays
    assert ok and out.split()[1] == "يَحُلُّ"


def test_worker_loads_the_model_from_the_named_folder(monkeypatch):
    import sys
    import types
    from muhawir import tashkeel_worker as w
    seen = {}

    class Fake:
        def __init__(self, **paths):
            seen.update(paths)

    monkeypatch.setitem(sys.modules, 'catt_tashkeel', types.SimpleNamespace(CATTEncoderOnly=Fake))
    monkeypatch.delenv('MUHAWIR_TASHKEEL_MODEL_DIR', raising=False)
    w.load_model(False)
    assert seen == {}  # no folder named: the package's own model
    monkeypatch.setenv('MUHAWIR_TASHKEEL_MODEL_DIR', '/models/eo')
    w.load_model(False)
    assert seen == {'encoder_path': '/models/eo/encoder.onnx', 'decoder_path': '/models/eo/decoder.onnx'}


def test_a_long_answer_is_read_in_rounds_and_each_sentence_only_once(monkeypatch, tmp_path):
    monkeypatch.setenv('MUHAWIR_TASHKEEL_PYTHON', str(fake_python(tmp_path, ECHO)))
    monkeypatch.setattr(t, '_BATCH', 1)
    monkeypatch.setattr(t, '_BUDGET', -1)  # out of time after the first batch
    texts = ['هذا كتاب.', 'وهذا قلم.', 'وتلك مسطرة.']
    first = t.diacritize(texts)
    assert first['complete'] is False and len(t._DONE) == 1
    second = t.diacritize(texts)
    assert second['complete'] is False and len(t._DONE) == 2
    third = t.diacritize(texts)
    assert third['complete'] is True and len(t._DONE) == 3 and third['texts'] == texts
    assert t.diacritize(texts)['complete'] is True  # all kept: nothing is read again

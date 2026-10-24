"""Display boundary tests: no downloads or real model calls."""
import subprocess
from types import SimpleNamespace
import pytest
from muhawir import tashkeel as t

@pytest.fixture(autouse=True)
def reset_limits():
    t._CALLS.clear()
    t._predict.cache_clear()


def test_mixed_content():
    text = 'هذا كتاب. English 12 اردو'
    result, ok = t.merge(text, 'هَذَا كِتَاب. English 12 اردو', ())
    assert ok and t.bare(result) == text
    assert result.endswith('English 12 اردو')

@pytest.mark.parametrize('prediction', ['هَذَا كِتَاب!', 'هذا كتاب جديد.', 'هذا  كتاب.', 'هدا كتاب.', '<script>هذا كتاب.</script>'])
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


def test_timeout(monkeypatch, tmp_path):
    python = tmp_path / 'python'; python.touch()
    monkeypatch.setenv('MUHAWIR_TASHKEEL_PYTHON', str(python))
    def fail(*args, **kwargs):
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        raise subprocess.TimeoutExpired('worker', 45)
    monkeypatch.setattr(t.subprocess, 'run', fail)
    with pytest.raises(t.Unavailable): t.diacritize(['هذا كتاب'])
    assert not t._SLOT.locked() and t._predict.cache_info().currsize == 0


def test_rejected_prediction(monkeypatch):
    monkeypatch.setattr(t, '_predict', lambda texts: ('كلام مختلف',))
    result = t.diacritize(['هذا كتاب'])
    assert result['texts'] == ['هذا كتاب'] and result['rejected'] == 1


def test_limits():
    with pytest.raises(ValueError): t.diacritize(['ا' * 6001])
    t._SLOT.acquire()
    try:
        with pytest.raises(t.Unavailable): t.diacritize(['سلام'])
    finally: t._SLOT.release()


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
    def fail(*args): raise t.Unavailable('offline')
    monkeypatch.setattr(t, 'diacritize', fail)
    assert api.post('/api/tashkeel', json={'texts':['سلام']}).status_code == 503


def test_omitted_final_period_restored():
    text = 'هذا كتاب.'
    output, ok = t.merge(text, 'هَذَا كِتَاب', ())
    assert ok and output == 'هَذَا كِتَاب.'
    assert t.bare(output) == text


def test_omitted_period_cannot_hide_changed_letter():
    assert t.merge('هذا كتاب.', 'هدا كتاب', ()) == ('هذا كتاب.', False)

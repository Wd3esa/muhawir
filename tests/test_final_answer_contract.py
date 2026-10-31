"""No live models. Structural regressions from the owner's two-turn screenshots."""
import json
from pathlib import Path
import pytest
from muhawir import generate as g
from muhawir.corpus import Corpus, Source, Passage
from muhawir.pipeline import Muhawir, Response
from muhawir.presentation import compact_provenance, lead_with_requested_text, retain_inline_qualification
from muhawir.verify import Claim

TEXT = 'عجبا لأمر المؤمن إن أمره كله خير وليس ذلك لأحد إلا للمؤمن إن أصابته سراء شكر فكان خيرا له وإن أصابته ضراء صبر فكان خيرا له'
P = Passage('m:test', 'sahih-muslim', 'Synthetic source', 'عن صهيب بن سنان رضي الله عنه قال: «' + TEXT + '».', 'hadith')
CORPUS = Corpus({'sahih-muslim': Source('sahih-muslim', 'صحيح مسلم', 'Synthetic test only')}, [P])
Q = 'حديث إن أصابته ضراء صبر وإن أصابته سراء شكر'
REQUEST = {'form': 'hadith_text', 'requirements': ['نص الحديث المقصود أولًا']}
QUOTE = Claim('روى مسلم في صحيحه عن صهيب أن النبي ﷺ قال: «' + TEXT + '».', (P.id,))
SUMMARY = Claim('يبين الحديث أن المؤمن يشكر عند السراء ويصبر عند الضراء.', (P.id,))


def engine(drafts, reviews, monkeypatch):
    seen = {'drafts': [], 'reviews': []}
    drafts, reviews = iter(drafts), iter(reviews)
    def call(system, user, schema):
        if system in (g.RELEVANCE_PROMPT, g.KIDS_REVIEW_PROMPT):
            seen['reviews'].append(user)
            return json.dumps(next(reviews), ensure_ascii=False)
        if 'claims' in schema.get('properties', {}):
            seen['drafts'].append(user)
            claims = next(drafts)
            return json.dumps({'claims': [{'text': c.text, 'passage_ids': c.passage_ids} for c in claims]}, ensure_ascii=False)
        return '{"absent": []}'
    gen = g.ModelGenerator([('fake', call)])
    monkeypatch.setattr(gen, 'check_support', lambda claims, *args: [True] * len(claims))
    return Muhawir(CORPUS, gen), seen


def test_requested_hadith_is_first_even_when_drafted_after_explanation(monkeypatch):
    m, seen = engine([[SUMMARY, QUOTE]], [{'verdict': 'yes'}], monkeypatch)
    result = m._write(Q, [P], 'youth', 'ar', False, {'answer_request': REQUEST})
    displayed, _, _ = m._display_claims(result.kept, CORPUS, 'youth', 'ar', False,
                                        {'answer_request': REQUEST}, Q)
    assert displayed[0] == QUOTE
    assert seen['reviews'][0].index(QUOTE.text) < seen['reviews'][0].index(SUMMARY.text)
    assert len(seen['drafts']) == len(seen['reviews']) == 1


def test_missing_requested_text_is_not_silently_replaced_by_a_summary(monkeypatch):
    m, seen = engine([[SUMMARY], [SUMMARY]], [], monkeypatch)
    result = m._write(Q, [P], 'youth', 'ar', False, {'answer_request': REQUEST})
    assert isinstance(result, Response) and result.status == 'unavailable'
    assert len(seen['drafts']) == 2 and not seen['reviews']


def test_reviewer_reads_the_same_shortened_text_that_will_be_published(monkeypatch):
    # The seventh item is an independent requested part; a pre-trim judge must not certify it.
    first = [Claim(f'توضيح مدعوم للمؤمن رقم {i}.', (P.id,)) for i in range(6)] + [QUOTE]
    m, seen = engine([first, [QUOTE, SUMMARY]], [{'verdict': 'partly', 'reason': 'نص الحديث غير موجود في الجواب المعروض'}, {'verdict': 'yes'}], monkeypatch)
    result = m._write('اذكر المعنى والنص', [P], 'youth', 'ar', False, {})
    assert QUOTE.text not in seen['reviews'][0]
    assert QUOTE.text in seen['reviews'][-1]
    # A shorter complete rewrite can replace a longer incomplete answer.
    assert result.kept == [QUOTE, SUMMARY] and not result.partial
    assert len(seen['drafts']) == len(seen['reviews']) == 2


def test_repeated_source_only_clause_is_removed_but_narrator_detail_is_kept():
    claims = [QUOTE, Claim('ومصدره صحيح مسلم.', (P.id,)),
              Claim('وقد أخرجه مسلم عن صهيب بن سنان رضي الله عنه.', (P.id,))]
    compact = compact_provenance(claims, CORPUS, 'ar')
    assert len(compact) == 2 and compact[0] == QUOTE
    assert compact[1].text == 'والراوي صهيب بن سنان رضي الله عنه.'
    assert compact_provenance(compact, CORPUS, 'ar') == compact


def test_deduplication_never_removes_a_grading_or_other_report():
    grading = Claim('ومصدره صحيح مسلم بإسناد ضعيف بحسب المصدر.', (P.id,))
    other = Passage('other', 'sahih-muslim', 'Other synthetic report', 'نص تجريبي آخر.', 'hadith')
    corpus = Corpus(CORPUS.sources, [P, other])
    distinct = Claim('ومصدره صحيح مسلم.', (other.id,))
    assert compact_provenance([QUOTE, grading, distinct], corpus, 'ar') == [QUOTE, grading, distinct]


def test_inline_non_establishment_notice_stays_with_its_unique_quotation():
    p = Passage('f', 's', 'Synthetic fiqh', 'وقد روي: «لا زكاة في مال حتى يحول عليه الحول». وسبب الاختلاف: أنه لم يرد في ذلك حديث ثابت.', 'fiqh')
    corpus = Corpus({'s': Source('s', 'Synthetic source', 'Test only')}, [p])
    c = Claim('روي: «لا زكاة في مال حتى يحول عليه الحول».', ('f',))
    fixed = retain_inline_qualification([c], corpus)
    assert 'لم يرد في ذلك حديث ثابت' in fixed[0].text
    assert retain_inline_qualification(fixed, corpus) == fixed
    unrelated = Claim('ورد قول آخر عن الشروط.', ('f',))
    assert retain_inline_qualification([unrelated], corpus) == [unrelated]


def test_ambiguous_multiple_reports_are_not_given_an_invented_grading():
    p = Passage('f', 's', 'Synthetic', '«نص رواية أولى في هذا الاختبار». «نص رواية ثانية في هذا الاختبار». لم يرد فيه حديث ثابت.', 'fiqh')
    corpus = Corpus({'s': Source('s', 'Synthetic', '')}, [p])
    c = Claim('«نص رواية أولى في هذا الاختبار».', ('f',))
    assert retain_inline_qualification([c], corpus) == [c]


def test_understanding_retains_semantic_request_in_one_call_and_old_replies_work():
    seen = []
    def call(system, user, schema):
        seen.append(user)
        return json.dumps({'question': Q, 'queries': ['عجبا لأمر المؤمن'], 'answer_request': REQUEST}, ensure_ascii=False)
    gen = g.ModelGenerator([('fake', call)])
    history = [{'role': 'assistant', 'text': SUMMARY.text}, {'role': 'user', 'text': Q}]
    request = gen.understand('اذكر الحديث أولا', history)['answer_request']
    assert request == REQUEST and len(seen) == 1 and SUMMARY.text in seen[0]
    prompt = g.build_user_prompt(Q, [P], 'youth', 'ar', False, answer_request=request)
    assert 'hadith_text' in prompt and REQUEST['requirements'][0] in prompt
    assert g.ModelGenerator([('old', lambda *args: '{"question":"س","queries":["س"]}')]).understand('س', [])['answer_request'] == {}


@pytest.mark.parametrize('request_data', [None, 'bad', {'form': 'bad', 'requirements': []}, {'form': 'hadith_text', 'requirements': 'bad'}])
def test_invalid_request_metadata_does_not_invent_a_deliverable(request_data):
    gen = g.ModelGenerator([('fake', lambda *args: json.dumps({'question': Q, 'queries': [Q], 'answer_request': request_data}))])
    assert gen.understand(Q, [])['answer_request'] == {}


def test_css_is_served_after_the_book_styles_without_model_calls(monkeypatch):
    from fastapi.testclient import TestClient
    from muhawir import server
    monkeypatch.setattr(server.engine.generator, 'generate', lambda *args, **kwargs: pytest.fail('no model calls'))
    client = TestClient(server.app)
    html = client.get('/').text
    assert html.index('/book-answer.css') < html.index('/surfaces.css')
    css = client.get('/surfaces.css')
    assert css.status_code == 200 and css.headers['content-type'].startswith('text/css')
    assert css.text == (server.ROOT / 'static/surfaces.css').read_text()


def test_short_authentic_text_form_is_not_rejected_by_an_arbitrary_word_limit():
    p = Passage('h', 's', 'Synthetic', '«كلمتان فقط»', 'hadith')
    c = Claim('«كلمتان فقط»', ('h',))
    assert lead_with_requested_text([SUMMARY, c], {'h': p}, REQUEST)[0] == c


def test_missing_parts_are_validated_indices_of_the_original_user_requirements(monkeypatch):
    request_data = {'form': 'explanation', 'requirements': ['المعنى', 'المثال المطلوب']}
    m, _ = engine([[SUMMARY], [SUMMARY]], [
        {'verdict': 'partly', 'missing_parts': [1, 999, True, 'invention', 1]},
        {'verdict': 'partly', 'missing_parts': [1]}], monkeypatch)
    result = m._write('اشرح المعنى مع مثال', [P], 'youth', 'ar', False, {'answer_request': request_data})
    assert result.partial and result.incomplete_parts == ('المثال المطلوب',)

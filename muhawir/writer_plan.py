"""Opt-in writer planning contract. Mapping evidence is not a semantic verdict."""
from copy import deepcopy
from dataclasses import dataclass
import json
import re

UNUSABLE_EVIDENCE_PLAN = "writer returned an unusable evidence plan"
OVERSIZED_DRAFT = "writer exceeded the draft size budget"

# Shared with final display: a candidate must fit before sentence checking,
# rather than relying on display truncation to hide an oversized answer.
STYLE_LIMITS = {"kids": (5, 7), "youth": (6, 10), "newcomer": (8, 10), "extended": (12, 16)}


class DraftTooLarge(ValueError):
    """A complete response whose raw claim/view counts exceed the candidate budget."""


def check_draft_size(data, style):
    """Count before parsing/deduplication; never truncate or normalize away a condition.

    Text and quotation lengths are not capped here. All original source texts,
    grades, authors and translations remain available to the writer and checker.
    """
    if not isinstance(data, dict):
        raise ValueError('draft must be an object')
    claims, views = data.get('claims', []), data.get('views', [])
    if not isinstance(claims, list) or not isinstance(views, list):
        raise ValueError('draft claims/views must be arrays')
    prose, items = STYLE_LIMITS.get(style, STYLE_LIMITS['youth'])
    # An invalid flag cannot disguise a count beyond even the list allowance.
    if len(claims) > items or len(views) > items:
        raise DraftTooLarge('raw claim/view count exceeds style budget')
    listing = data.get('as_list', False)
    if type(listing) is not bool:
        raise ValueError('draft list flag must be boolean')
    if len(claims) > (items if listing else prose):
        raise DraftTooLarge('raw claim/view count exceeds style budget')


def bounded_schema(schema, style):
    """Advertise upper bounds, without depending on provider schema enforcement."""
    out = deepcopy(schema)
    items = STYLE_LIMITS.get(style, STYLE_LIMITS['youth'])[1]
    for field in ('claims', 'views'):
        out['properties'][field]['maxItems'] = items
    return out


def planned_prompt(base, style):
    """Use one plan format in instructions, examples and final output template.

    Preserve the existing religious/source rules and example claims verbatim;
    change only planning, evidence aggregation and the raw output budget.
    """
    old = '6. فكّر قبل أن تكتب، في الحقل plan (لا يراه السائل): ما نوع السؤال، وأي المقاطع تجيب عنه وماذا يقول كل منها باختصار، وترتيب الشرح.'
    aggregate = 'واجمع المقاطع المتعلقة كلها (آية وحديث وكلام عالم) في شرح واحد متصل يُقرأ كحديث طبيعي.'
    if base.count(old) != 1 or base.count(aggregate) != 1:
        raise ValueError('writer prompt changed; review the planned format before sending')
    prompt = base.replace(old, '6. خطط في plan بربط كل مطلب مرقم بأدلته اللازمة فقط؛ لا تكتب ملخصًا لكل مقطع.')
    prompt = prompt.replace(aggregate, 'واجمع ما يلزم للإجابة عن مطالب السؤال من المقاطع في شرح متصل؛ لا تسرد كل ما في المصادر ولا تعِد المعنى بتعددها.')
    # The two examples and final template are JSON objects. Replacing only their
    # plan values keeps all claim texts/citations, grades and other fields intact.
    decoder, spans = json.JSONDecoder(), []
    for match in re.finditer(r'\{"plan": ', prompt):
        obj, length = decoder.raw_decode(prompt[match.start():])
        if not isinstance(obj.get('plan'), str):
            raise ValueError('unexpected plan example format')
        ids = list(dict.fromkeys(pid for c in obj.get('claims', []) + obj.get('views', [])
                                 for pid in c.get('passage_ids', [])))
        obj['plan'] = [{'requirement': 0, 'passage_ids': ids}]
        spans.append((match.start(), match.start() + length, json.dumps(obj, ensure_ascii=False)))
    if len(spans) != 3:
        raise ValueError('review all writer examples before changing their plan format')
    for start, end, replacement in reversed(spans):
        prompt = prompt[:start] + replacement + prompt[end:]
    prose, items = STYLE_LIMITS.get(style, STYLE_LIMITS['youth'])
    budget = (f'حد المسودة قبل فحصها: claims حتى {prose} جمل في النثر، أو {items} عناصر عند as_list=true، '
              f'وviews حتى {items} أقوال لازمة فقط. هذا حد أقصى لا عدد مطلوب بلوغه. '
              'لا تختصر بحذف شرط أو نفي أو قائل أو درجة رواية، ولا تجعل كل مصدر فقرة مستقلة. '
              'اكتب كل معنى لازم مرة واحدة مع مراجع أدلته، ثم أغلق JSON عند اكتمال المطالب؛ '
              'لا تحشد أدلة أو أمثلة إضافية لملء الرد. لا تخفِ مطلبًا لم تستطع استيفاءه ضمن الحد ولا تدّع اكتماله.')
    return prompt + '\n\n' + PLAN_INSTRUCTION + '\n\n' + budget


@dataclass(frozen=True)
class EvidenceTask:
    requirement: int
    passage_ids: tuple[str, ...]


def requirements_for(question, request):
    values = request.get('requirements', []) if isinstance(request, dict) else []
    if not isinstance(values, list):
        values = []
    return tuple(value.strip() for value in values if isinstance(value, str) and value.strip())[:6] or (question,)


def planned_schema(base, count, passage_ids):
    """Change only the old free-text plan, preserving all claim/view constraints."""
    schema = deepcopy(base)
    schema['properties']['plan'] = {
        'type': 'array', 'minItems': count, 'maxItems': count,
        'items': {'type': 'object',
                  'properties': {'requirement': {'type': 'integer', 'minimum': 0, 'maximum': count - 1},
                                 'passage_ids': {'type': 'array', 'uniqueItems': True,
                                                 'items': {'type': 'string', 'enum': list(passage_ids)}}},
                  'required': ['requirement', 'passage_ids'], 'additionalProperties': False}}
    return schema


def read_plan(data, count, passages, claims):
    """Every requirement needs one row; citations must be given and actually planned.

    An empty source list records an unresolved part, not proof that no source
    exists. The final answer still passes the ordinary semantic/coverage checks.
    """
    if not isinstance(data, dict) or not isinstance(data.get('plan'), list) or len(data['plan']) != count:
        return None
    rows, seen, selected = [], set(), set()
    allowed = {passage.id for passage in passages}
    for row in data['plan']:
        if not isinstance(row, dict) or set(row) != {'requirement', 'passage_ids'}:
            return None
        index, ids = row['requirement'], row['passage_ids']
        if type(index) is not int or not 0 <= index < count or index in seen:
            return None
        if not isinstance(ids, list) or any(not isinstance(pid, str) or pid not in allowed for pid in ids):
            return None
        if len(set(ids)) != len(ids):
            return None
        seen.add(index)
        selected.update(ids)
        rows.append(EvidenceTask(index, tuple(ids)))
    if any(not claim.passage_ids or not set(claim.passage_ids) <= selected for claim in claims):
        return None
    return tuple(sorted(rows, key=lambda row: row.requirement))


PLAN_INSTRUCTION = """صيغة تخطيط تجريبية داخل نداء الكتابة نفسه:
حقل plan مصفوفة في التعليمات والأمثلة والجواب. لكل مطلب مرقم صف واحد
بالشكل {"requirement": 0, "passage_ids": ["معرف مقطع معطى"]}. لا تضف حكما أو نصا في الخطة.
رتّب الإجابة على مطالب السؤال، ولا تكرر المعنى لمجرد اشتراك أكثر من مطلب في الدليل.
الخطة ليست لائحة لتلخيص جميع المقاطع، ولا يلزم ذكر كل مصدر فيها في فقرة مستقلة.
إن لم تجد دليل مطلب فاترك قائمته فارغة؛ هذا لا يثبت عدم وجود مصدر ولا يجيز اختراع جواب.
استعمل في الجمل معرّفات من خطتك فقط؛ لا تحذف درجة الرواية أو شرط القول أو قائله للتقصير.
الخطة ليست اعتمادا لصحة المعنى، وستمر كل الجمل بفحص السند والتغطية المعتاد."""

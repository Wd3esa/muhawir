"""Opt-in writer planning contract. Mapping evidence is not a semantic verdict."""
from copy import deepcopy
from dataclasses import dataclass

UNUSABLE_EVIDENCE_PLAN = "writer returned an unusable evidence plan"


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
حقل plan هنا مصفوفة بدل النص الحر المستعمل في أمثلة الشرح. لكل مطلب مرقم صف واحد
بالشكل {"requirement": 0, "passage_ids": ["معرف مقطع معطى"]}. لا تضف حكما أو نصا في الخطة.
رتّب الإجابة على مطالب السؤال، ولا تكرر المعنى لمجرد اشتراك أكثر من مطلب في الدليل.
إن لم تجد دليل مطلب فاترك قائمته فارغة؛ هذا لا يثبت عدم وجود مصدر ولا يجيز اختراع جواب.
استعمل في الجمل معرّفات من خطتك فقط؛ لا تحذف درجة الرواية أو شرط القول أو قائله للتقصير.
الخطة ليست اعتمادا لصحة المعنى، وستمر كل الجمل بفحص السند والتغطية المعتاد."""

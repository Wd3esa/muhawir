"""Offline recording integrity and source recovery; synthetic content only."""
import copy
import hashlib
import json
from pathlib import Path
import stat

import pytest

from scripts import prepare_joint_review as prep


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture_recording(tmp_path, bad_quote=False):
    text = "نص تجريبي يذكر الصناديق الزرقاء فقط."
    source = {"id": "p1", "source_id": "s", "source": "مصدر اصطناعي",
              "location": "موضع تجريبي", "text_sha256": digest(text)}
    user = f"المقاطع:\n\n[p1] (فقه — موضع تجريبي)\n{text}\n\n\n\nأسلوب الشرح: تجريبي"
    cases = [{"id": f"T{i}", "question": "ما لون الصناديق؟", "style": "youth",
              "user": user, "user_sha256": digest(user), "evidence": [copy.deepcopy(source)],
              "order": ["current", "candidate"] if i % 2 == 0 else ["candidate", "current"]}
             for i in range(4)]
    plan = {"scope": "exploratory_draft_only_not_release", "cases": cases,
            "systems": {"current": "fake", "candidate": "fake plus"},
            "system_sha256": {"current": digest("fake"), "candidate": digest("fake plus")}}
    rows = [{"kind": "plan", "plan": plan,
             "plan_sha256": digest(json.dumps(plan, ensure_ascii=False, indent=2))}]
    for case in cases:
        for variant in case["order"]:
            rows.append({"kind": "attempt_started", "case_id": case["id"], "variant": variant})
            draft = {"plan": "test", "abstain": False, "as_list": False, "claims": [{
                "text": "«اقتباس غير موجود»" if bad_quote and case["id"] == "T1" else "الصناديق زرقاء.",
                "passage_ids": ["p1"], "section": "", "label": ""}], "views": [], "follow_up": ""}
            rows.append({"kind": "response", "case_id": case["id"], "variant": variant,
                         "response": {"id": case["id"] + variant, "choices": [{"finish_reason": "stop",
                             "message": {"content": json.dumps(draft, ensure_ascii=False)}}]}})
    rows.append({"kind": "finished", "status": "completed_draft_only_no_quality_verdict"})
    path = tmp_path / "recording.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    return path, rows


def test_frozen_inputs_recover_sources_and_preserve_locally_rejected_case(tmp_path):
    path, _ = fixture_recording(tmp_path, bad_quote=True)
    before = path.read_bytes()
    data = prep.prepare_recording(path)
    assert path.read_bytes() == before
    assert len(data["requests"]) == 4 and data["human_reference_status"] == "pending"
    blocked = data["requests"][1]
    assert blocked["request"] is None and blocked["local_rejections"]
    assert blocked["status"] == "blocked_by_existing_code_check"
    prepared = data["requests"][0]
    user = json.loads(prepared["request"]["user"])
    assert user["passages"][0]["text"] == "نص تجريبي يذكر الصناديق الزرقاء فقط."
    assert user["passages"][0]["source_name"] == "مصدر اصطناعي"
    assert "candidate" not in {r["variant"] for r in data["requests"]}


@pytest.mark.parametrize("change", [
    lambda rows: rows[0]["plan"]["cases"][0].update(question="changed"),
    lambda rows: rows.pop(),
    lambda rows: rows.insert(2, rows[1]),
    lambda rows: rows[2]["response"]["choices"][0].update(finish_reason="length"),
    lambda rows: rows[2].update(kind="failure"),
])
def test_changed_or_incomplete_recording_rejected(tmp_path, change):
    path, rows = fixture_recording(tmp_path)
    change(rows)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    with pytest.raises(ValueError):
        prep.prepare_recording(path)


def test_user_hash_cannot_hide_changed_source_text(tmp_path):
    _, rows = fixture_recording(tmp_path)
    case = rows[0]["plan"]["cases"][0]
    case["user"] = case["user"].replace("الزرقاء", "الخضراء")
    case["user_sha256"] = digest(case["user"])
    with pytest.raises(ValueError, match="changed evidence text"):
        prep.corpus_from_case(case)


def test_private_output_cannot_overwrite_or_escape_measurement_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    path = tmp_path / "data/private/measurement/inputs.json"
    prep.save_private({"private": True}, path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        prep.save_private({}, path)
    with pytest.raises(ValueError):
        prep.save_private({}, tmp_path / "public.json")
    outside = tmp_path / "outside"
    outside.mkdir()
    (path.parent / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        prep.save_private({}, path.parent / "link/leak.json")

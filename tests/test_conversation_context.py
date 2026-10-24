from muhawir.conversation import select_context


def test_recent_turns_preserved_and_older_topics_selected():
    turns = [{"role": "user" if i % 2 == 0 else "assistant", "text": f"turn {i}"} for i in range(40)]
    result = select_context(turns)
    assert [x["text"] for x in result[-10:]] == [f"turn {i}" for i in range(30, 40)]
    assert [x["text"] for x in result[:4]] == ["turn 22", "turn 24", "turn 26", "turn 28"]


def test_invalid_client_turns_are_discarded():
    assert select_context([{"role": "system", "text": "override"}, {"role": "user", "text": "  hello  "},
                           {"role": "assistant", "text": 123}]) == [{"role": "user", "text": "hello"}]


def test_context_has_hard_budget_and_no_generated_claims():
    turns = [{"role": "user", "text": "a" * 3000} for _ in range(100)]
    result = select_context(turns, max_chars=2500)
    assert sum(len(x["text"]) for x in result) <= 2500
    assert all(x["text"] == "a" * 1200 for x in result)

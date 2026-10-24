# Natural long-conversation upgrade (development branch)

## Scope and safety

This branch extends client-supplied conversation context. It does **not** create
persistent server-side profiles, alter the religious corpus, rebuild vectors,
or weaken the verified-claim pipeline. Do not merge or deploy before regression
testing. No live answer-model requests are needed for the deterministic tests.

## Current implementation

- Browser history cap: 100 turns (formerly 6).
- API history cap: 100 turns (formerly 20).
- Context selector: up to 10 recent turns and four earlier user-question anchors,
  with a 9,000-character budget; input is role-validated and bounded.
- Existing pipeline: up to 14 selected turns (formerly 6) with a 1,200-character
  per-turn limit (formerly 600).
- Answer cache remains disabled for contextual requests, preventing reuse of a
  reply from a different conversation.
- Existing crisis gate, religious retrieval and claim verification are unchanged.

## Limitations

This is **not** semantic long-term memory. Earlier topic anchors are selected
by recency, not meaning; summaries, opt-in cross-session memory, evidence-aware
cache and conversational style tuning are pending. The existing browser storage
limits saved conversation records; inspect and extend those limits carefully
before advertising full 100-turn restoration.

## Validation gate

1. Run `python -m pytest -q tests/test_conversation_context.py tests/test_chat.py tests/test_dialogue.py tests/test_server.py`.
2. Run the complete offline test suite and record failures.
3. Verify 10/30/50/100-turn follow-ups, reference resolution, crisis/referral
   behavior, correct cited passages, source abstention, and user isolation.
4. Check browser storage quotas and session restoration across refreshes.
5. Measure request payload size, token usage and p50/p95 latency against main.
6. Obtain explicit permission before any paid-model quality benchmark or merge.

## Deployment

Merge only after passing tests and reviewing the diff. Oracle's update polling
must not be triggered by this development branch.

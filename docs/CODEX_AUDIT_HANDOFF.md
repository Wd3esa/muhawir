# Codex audit handoff for Claude Code

Audit date: 2026-10-06. Reviewed `main` at `3a7a94d`. This document is an audit handoff only; no source changes are included in this branch. Re-check the current HEAD and existing work before changing files.

## Reproduced findings, in priority order

1. **Verse translation bypasses approved sources and verification.** In `muhawir/pipeline.py:653-661`, a model-provided `translate` value takes a direct translation path and returns `TRANSLATED` with no source cards or second reading. With a fake model, asking to translate `قُلْ هُوَ اللَّهُ أَحَدٌ` returned arbitrary incorrect text as `translated`, `sources=[]`. The `TRANSLATE_PROMPT` explicitly permits an ayah or part of one. Route religious text through approved Quran translation (and cite its passage); preserve ordinary word/phrase translation where appropriate. Add a negative test for an incorrect verse translation.

2. **First-person personal cases evade the fixed referral gate.** `muhawir/classify.py:45-50` returns `None` for `فاتتني الصلاة، ماذا أفعل؟`, `نسيت صلاة الفجر، كيف أقضيها؟`, `I missed a prayer. What should I do?`, and `I cannot fast this Ramadan. What is required of me?`. In a fake-generator reproduction, a supported claim then returned `answered` with no referral. Model prompts may sometimes catch the intent, but the fixed gate does not enforce it. Extend Arabic/English patterns and add an independent final referral safeguard for individual circumstances; test paraphrases and distinguish general questions.

3. **Live Action can be green when every question fails.** `.github/workflows/live-test.yml` runs `scripts/try_questions.py`. Its per-question exception handler records `ERROR:` and continues; the script exits 0. Reproduction: `python scripts/try_questions.py --url http://127.0.0.1:9 --out /tmp/muhawir-audit-failed-live.txt` produced 32 errors and exit code 0. The wake loop also ends with success after all health attempts fail, and the Action never runs the quantitative `scripts/evaluate.py` or produces `docs/RESULTS.md`. Fail on health/request errors and required thresholds, upload both machine-readable and human reports, and mind model/Actions quota before a full live run.

4. **A rebuilt vector index can keep old text embeddings.** `muhawir/vectors.py:_start_checkpoint` resumes from the completed index when IDs and model match, without checking embedded content. Calling `_embed_corpus` twice with the same ID and model but changed passage text resulted in `resumed=1` and only one total embedding call; the changed text was never embedded. This can persist across normal rebuilds of the corpus. Store/compare a content fingerprint of each passage (and model/dimensions) in checkpoints, index metadata, and deployment validation; invalidate changed rows. Do not simply compare IDs.

5. **Malformed primary model output skips fallback.** `muhawir/generate.py:860-885` returns empty claims immediately after `parse_draft` fails. Direct reproduction with primary `not json` and a valid backup called only the primary. Continue to the next provider for unusable output; preserve an intentional, well-formed abstention. Test both cases.

6. **Retry model outage is mislabeled as missing sources.** `muhawir/pipeline.py:575-589` discards a `Response(UNAVAILABLE)` from the second search. In a reproduction where first search had no passages, second search found one, and generation failed, the result was `abstained` rather than `unavailable`. Propagate the outage when there is no earlier usable answer.

## Other findings requiring focused verification

7. `/tts` accepts arbitrary caller text up to 3000 characters and stores audio on disk; the browser sends only Muhawir answer text, but server enforcement is absent. Reproduced direct synthesis of an arbitrary string with mocked Azure. Bind requests to server-issued answers, set retention/size limits for `data/tts-cache`, and assess the 20/min IP limit behind the production reverse proxy. Browser fallback `pickVoice` uses language but ignores the selected male/female option. Files: `muhawir/server.py:79-113`, `muhawir/tts.py`, `muhawir/static/index.html:1842-1846`.

8. `/api/health` always returns `ok: true` even with the synthetic corpus. In this local checkout it reported `{'ok': True, 'generator': 'extractive', 'synthetic': True, 'passages': 5}`. Production readiness should require the real database while development can opt into synthetic mode. File: `muhawir/server.py:116-119`.

9. A relevance verdict of `partly` triggers an answer rewrite, but the accepted rewrite is not judged again (`muhawir/pipeline.py:553-573`). Existing `tests/test_dialogue.py` explicitly expects one relevance judgment. Decide whether the new answer needs a final relevance check, while retaining bounded model calls.

10. Static lint finds one unused import, `collections.Counter` in `scripts/evaluate.py:28`. Lower priority.

## Verification and scope

- Full local suite: **334 passed, 1 skipped**; Python package/test pyflakes clean; script pyflakes finding above. Inline JavaScript syntax and shell syntax checks passed.
- The repository checkout contains only `data/synthetic_corpus.json`; no real corpus database, Azure key, or live model was available for this audit. The audit does not establish religious correctness or live-site behavior. Conduct live evaluation and human source review before claiming those outcomes.
- Suggested implementation order: source/translation and personal referral; fail-the-Action behavior; vector freshness; model fallback and truthful status; TTS and readiness; low-impact polish.

Please verify each item against the latest branch, identify false positives or prior fixes, propose focused changes and regression tests, and coordinate edits with Codex. Do not merge into `main` before the owner reviews the patch.

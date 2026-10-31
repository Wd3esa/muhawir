"""Explicit complete replies for fake providers; never repairs production output."""


def support_reply(verdicts, witness, *, on_topic=None, missing=None, evidence=None):
    problems = ['none' if value is True else 'addition' if value is False else value for value in verdicts]
    size = len(problems)
    return {'problem': problems,
            'on_topic': [True] * size if on_topic is None else on_topic,
            'missing': [[] for _ in range(size)] if missing is None else missing,
            'evidence': [witness] * size if evidence is None else evidence}

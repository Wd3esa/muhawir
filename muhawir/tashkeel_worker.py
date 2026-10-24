"""Runs in the isolated, already-tested CATT environment; JSON in/out, no LLM calls."""
import contextlib
import json
import sys


def main():
    texts = json.load(sys.stdin)
    if not isinstance(texts, list) or len(texts) > 30 or not all(isinstance(t, str) for t in texts) or sum(map(len, texts)) > 6000:
        raise ValueError('Invalid input')
    with contextlib.redirect_stdout(sys.stderr):
        from catt_tashkeel import CATTEncoderOnly
        model = CATTEncoderOnly()
        output = [model.do_tashkeel_batch([text], batch_size=1, verbose=False)[0] if text.strip() else text for text in texts]
    json.dump(output, sys.stdout, ensure_ascii=False)


if __name__ == '__main__':
    main()

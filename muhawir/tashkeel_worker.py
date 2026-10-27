"""Long-lived worker in the isolated, already-tested CATT environment; no LLM calls.

Protocol, one JSON per line: it prints {"ready": true} once the model is loaded, then for each input line (a JSON
list of texts) prints one line (a JSON list of vowelled texts). It exits when its input ends."""
import contextlib
import json
import os
import sys


# Letters (Buckwalter, as the model reads them) that may end a word with no mark: a long vowel (alef, alef maqsura,
# waw, ya) or the madd alef. Every other letter at the end of a word carries a vowel or a sukun in fully vowelled text.
UNMARKED_ENDINGS = frozenset("AYwy|")


def forbid_unmarked_endings(logits, input_ids, letters, no_mark_id):
    """The model's scores with «no mark» made impossible on the last letter of every word that cannot end without one.
    The model then gives the best mark it has for that letter instead of leaving it bare («قَوْله» for «قَوْلِهِ»).
    `logits` is (batch, length, classes), `input_ids` (batch, length), `letters` the model's letter list."""
    import numpy as np
    out = np.array(logits, copy=True)
    for row in range(input_ids.shape[0]):
        for pos in range(input_ids.shape[1]):
            letter = letters[input_ids[row, pos]]
            if len(letter) != 1 or letter == " " or letter in UNMARKED_ENDINGS:
                continue  # a special token, a space, or a letter that may end bare
            following = letters[input_ids[row, pos + 1]] if pos + 1 < input_ids.shape[1] else " "
            if following in (" ", "<PAD>", "<EOS>"):
                out[row, pos, no_mark_id] = -np.inf
    return out


def model_paths() -> dict:
    """Where the model files are: the folder named by MUHAWIR_TASHKEEL_MODEL_DIR (an encoder.onnx and a decoder.onnx, as
    the fine-tuned model is exported), else the package's own released model."""
    folder = os.environ.get("MUHAWIR_TASHKEEL_MODEL_DIR")
    if not folder:
        return {}
    return {"encoder_path": os.path.join(folder, "encoder.onnx"), "decoder_path": os.path.join(folder, "decoder.onnx")}


def load_model(endings: bool):
    """The encoder-only model; with `endings` it is forced to mark the end of every word that needs one."""
    from catt_tashkeel import CATTEncoderOnly
    paths = model_paths()
    if not endings:
        return CATTEncoderOnly(**paths)

    import numpy as np

    class EndingsEncoderOnly(CATTEncoderOnly):
        def _process_batch(self, batch_input_ids):
            ids = batch_input_ids[:, 1:-1]
            logits = self._run_decoder(self._run_encoder(ids))
            none = self.tokenizer.tashkeel_map[self.tokenizer.no_tashkeel_tag]
            predictions = np.argmax(forbid_unmarked_endings(logits, ids, self.tokenizer.letters, none), axis=-1)
            return self._apply_space_mask(predictions, ids)

    return EndingsEncoderOnly(**paths)


def main():
    """Load the model once, then answer each input line (a JSON list of texts) with one line of vowelled texts."""
    out = sys.stdout
    sys.stdout = sys.stderr  # anything the library prints must not reach the protocol channel
    with contextlib.redirect_stdout(sys.stderr):
        model = load_model(os.environ.get('MUHAWIR_TASHKEEL_ENDINGS') == '1')
    out.write('{"ready": true}\n')
    out.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            texts = json.loads(line)
            if not isinstance(texts, list) or len(texts) > 300 or not all(isinstance(t, str) for t in texts) \
                    or sum(map(len, texts)) > 12000:
                raise ValueError('Invalid input')
            with contextlib.redirect_stdout(sys.stderr):
                result = [model.do_tashkeel_batch([t], batch_size=1, verbose=False)[0] if t.strip() else t
                          for t in texts]
        except Exception as exc:  # noqa: BLE001 - one bad request must not end the worker
            result = {"error": str(exc)[:200]}
        out.write(json.dumps(result, ensure_ascii=False) + "\n")
        out.flush()


if __name__ == '__main__':
    main()

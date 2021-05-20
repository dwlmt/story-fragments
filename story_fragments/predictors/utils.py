import re

import more_itertools
from blingfire import text_to_sentences

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")


def input_to_passages(inputs, sentence_batch_size: int = 6, sentence_label_size: int = 6,
                      sentence_step_size: int = 6, max_passages: int = None, prefill: bool = False,
                      previous_text=None):
    labels = []

    # print(f"Inputs: {inputs}")
    if "sentences" in inputs and len(inputs["sentences"]) > 0:
        sentences = inputs["sentences"]
        extracted_sentences = []
        for s in sentences:
            if isinstance(s, dict):
                extracted_sentences.append(s["text"])
        if len(extracted_sentences) > 0:
            sentences = extracted_sentences

    elif "text" in inputs and len(inputs["text"]) > 0:
        text = _RE_COMBINE_WHITESPACE.sub(" ", inputs["text"]).strip()
        sentences = text_to_sentences(text).split('\n')
    elif "passages" in inputs and len(inputs["passages"]) > 0:
        passages = []
        labels = []
        for p in inputs["passages"]:
            if isinstance(p, dict):
                passages.append(p["text"])

                if "label" in p:
                    labels.append(p["label"])
            else:
                passages.append(p)

        sentences = None
    else:
        raise ValueError("Input text or sentences must be provided.")

    if sentences is not None:

        # Need to prefill with blank sentences to support a sliding window.
        if prefill:

            if previous_text:

                sentences = previous_text[-sentence_batch_size] + sentences + [" <PLACEHOLDER> "] * (
                    sentence_batch_size)

            else:
                sentences = [" <PLACEHOLDER> "] * sentence_batch_size + sentences + [" <PLACEHOLDER> "] * (
                    sentence_batch_size)

        sentences = list(more_itertools.windowed(sentences, n=sentence_batch_size + sentence_label_size, fillvalue=" ",
                                                 step=sentence_step_size))

        passages = [{"id": f"{i}", "seq_num": i, "text": " ".join(s[: sentence_batch_size]),
                     "label": " ".join(s[-sentence_label_size:]), "prompt": True} for i, s in enumerate(sentences)]

        if sentence_step_size < sentence_batch_size or sentence_step_size < sentence_label_size:
            for p, s in zip(passages, sentences):
                # print(f"{p}, {s}")
                sentences_offset = s[sentence_batch_size - sentence_step_size:sentence_batch_size]
                p["text_offset"] = " ".join(sentences_offset)
    else:
        passages = [{"id": f"{i}", "seq_num": i, "text": s, "prompt": True} for i, s in enumerate(passages)]

    if labels is not None and len(labels) > 0:
        if len(labels) < len(passages):
            labels += [" "] * (len(labels) - len(passages))
        for p, l in zip(passages, labels):
            p["label"] = l

    if max_passages is not None:
        passages = passages[:max_passages]

    return passages

import re
from collections import deque

import more_itertools
from blingfire import text_to_sentences
from datasets import logger

from story_fragments.data.contraction_utils import CONTRACTIONS_LIST


_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

def interleave_examples(reader, batch_size: int = 1, input_size: int = 1,
                        label_size: int = 1, step_size: int = 1,
                        dummy: bool = False,
                        dummy_max_examples: int = 10000,
                        contractions=False):
    """ Interleaves epsiodes examples, processes and returns as as dict.

    Args:
        reader: The JSON files reader.
        batch_size (int):  Number of parallel inputs to interleave.
        input_size (int): Size in sentences of the 'input_text' field.
        label_size (int):  Size in sentences of the 'target_text' field.
        step_size (int):  Sliding window step.
    """
    # Iterate over a batch of the input_text.

    episodes_example_list = deque([])

    id_counter = 0
    for episode in reader:

        logger.info(f"{episode}")
        text = f"{episode['text']}"

        def cleanup_text(text):
            if contractions:
                for e, r in CONTRACTIONS_LIST:
                    text = text.replace(e, r)

            text = text.replace("\t", " ")
            text = text.replace("\n", " ")

            if text.startswith('"'):
                text = text[1:]
            if text.endswith('"'):
                text = text[:-1]

            text = _RE_COMBINE_WHITESPACE.sub(" ", text).strip()

            return text

        sentences = text_to_sentences(text).split('\n')

        sentences = [cleanup_text(s) for s in sentences]


        # Skip episodes that are too short for the window.
        if len(sentences) <= input_size + label_size + step_size:
            continue

        windowed_sentences = list(
            more_itertools.windowed(sentences, n=input_size + label_size, fillvalue=" ", step=step_size))

        example_list = deque([])

        for i, window in enumerate(windowed_sentences):
            input_text = window[: input_size]
            label_text = window[-label_size:]

            if 'id' in episode:
                id = f"{episode['id']}"
            else:
                id = id_counter

            example = {
                "id": f"{id}-{i}",
                "episode_id": id,
                "episode_seq_num": i,
                "title": f"{episode['title']}",
                "text": " ".join(input_text),
                "label": " ".join(label_text),
                "episode_done": False if i < len(windowed_sentences) - 1 else True,
                "episode_begun": True if i == 0 else False
            }

            example_list.append(example)

        id_counter += 1

        episodes_example_list.append(example_list)

    batch_list = []

    example_counter = 0
    # Keep going while episodes left.
    while len(batch_list) > 0 or len(episodes_example_list) > 0:

        # Add to the batch of episodes up to the batch size.
        while len(batch_list) < batch_size and len(episodes_example_list) > 0:
            batch_list.append(episodes_example_list.popleft())

        # Yield the next example from each episode.
        for episode in batch_list:
            example = episode.popleft()
            # print(example)
            yield example
            example_counter += 1

            if dummy and example_counter >= dummy_max_examples:
                return

        # Remove empty episodes from the batch.
        batch_list = deque([e for e in batch_list if len(e) > 0])

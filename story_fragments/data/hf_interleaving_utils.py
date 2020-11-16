from collections import deque

import more_itertools
from blingfire import text_to_sentences


def interleave_examples(reader, batch_size: int = 1, input_size: int = 1,
                        label_size: int = 1, step_size: int = 1, dummy: bool = False, dummy_max_examples: int = 10000):
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

    for episode in reader:

        sentences = text_to_sentences(f"{episode['text']}").split('\n')

        # Skip episodes that are too short for the window.
        if len(sentences) <= input_size + label_size + step_size:
            continue

        windowed_sentences = list(
            more_itertools.windowed(sentences, n=input_size + label_size, fillvalue=" ", step=step_size))

        example_list = deque([])

        for i, window in enumerate(windowed_sentences):
            input_text = window[: input_size]
            label_text = window[-label_size:]
            example = {
                "id": f"{episode['id']}-{i}",
                "episode_id": f"{episode['id']}",
                "episode_seq_num": i,
                "title": f"{episode['title']}",
                "text": " ".join(input_text),
                "label": " ".join(label_text),
                "episode_done": False if i < len(windowed_sentences) - 1 else True,
                "episode_begun": True if i == 0 else False
            }

            example_list.append(example)

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

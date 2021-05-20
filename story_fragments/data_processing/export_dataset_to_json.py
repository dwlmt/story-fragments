import copy
import os

import fire
from datasets import load_dataset
from jsonlines import jsonlines


class ExportDatasetToJson(object):
    ''' Outputs datasets to a nested format.
    '''

    def export(self,
               script_path: str,
               dataset_name: str,
               output_file: str,
               split: str = "train",
               nested: bool = False
               ):

        output_list = []

        from pathlib import Path
        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

        dataset = load_dataset(script_path, dataset_name, split=split)

        episode_passages = []
        start_of_episode = None

        seq_counter = 0

        for example in dataset:

            if not nested:
                output_list.append(example)
            else:
                if "episode_begun" in example and example["episode_begun"] == True:
                    episode = copy.deepcopy(example)
                    episode_passages.append(
                        {"text": example["text"], "seq_num": seq_counter, "label": example["label"]})
                    del episode["text"]
                    del episode["episode_seq_num"]
                    del episode["episode_begun"]
                    del episode["episode_done"]
                    del episode["negative_labels"]
                    start_of_episode = episode
                elif "episode_done" in example and example["episode_done"] == True:
                    episode_passages.append(
                        {"text": example["text"], "seq_num": seq_counter, "label": example["label"]})
                    start_of_episode["passages"] = episode_passages
                    output_list.append(start_of_episode)
                    episode_passages = []
                    seq_counter = 0
                else:
                    episode_passages.append(
                        {"text": example["text"], "seq_num": seq_counter, "label": example["label"]})
                    seq_counter += 1

        print(f"Output JSON: {output_list[-1]}")

        with jsonlines.open(f'{output_file}', mode='w') as writer:
            for output in output_list:
                writer.write(output)


if __name__ == '__main__':
    fire.Fire(ExportDatasetToJson)

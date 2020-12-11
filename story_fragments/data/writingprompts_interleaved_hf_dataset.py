import os
from typing import Optional

import datasets
from datasets.info import SupervisedKeysData
from jsonlines import jsonlines

from story_fragments.data.hf_interleaving_utils import interleave_examples


_VERSION = datasets.Version("1.0.0")

_CITATION = """\
@inproceedings{fan-etal-2018-hierarchical,
    title = "Hierarchical Neural Story Generation",
    author = "Fan, Angela  and
      Lewis, Mike  and
      Dauphin, Yann",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P18-1082",
    doi = "10.18653/v1/P18-1082",
    pages = "889--898",
}
"""

_DESCRIPTION = """\
 The WritingPrompts dataset is over 300K short stories collected from the reddit forum /r/WritingPrompts/ .
 Each story is a creative writing exercise following a prompt.
"""

_URL = "https://drive.google.com/uc?export=download&id=1b8Q4_t2D0IKUXmlTY8sNV5ga0cB8kdvp"

_DOWNLOAD_NUM_BYTES = 250330124
_DOWNLOAD_CHECKSUM = "1c8886bec4948a77d16255f6e80178ae65dc4d28c24f543d5b2f6c7aaa057238"


class WritingPromptsInterleavedHfDatasetConfig(datasets.BuilderConfig):

    def __init__(self,
                 data_url: str,
                 data_download_num_bytes: Optional[int],
                 data_download_checksum: Optional[str],
                 input_size: int = 1,
                 target_size: int = 1,
                 step_size: int = 1,
                 batch_size: int = 32,
                 dummy: bool = False,
                 add_negative_examples: bool = False,
                 **kwargs):
        """ Generic config for reading a dataset in a interleaved or round robin fashion.

        Args:
            data_url (str): The url for the compressed jsonl file.
            data_download_num_bytes (int): Number of bytes of the datafile.
            data_download_checksum (str): SHA-256 checksum for the data file.
            input_size (int): Size in sentences of the context input_text to condition on.
            target_size (int): Size in sentences of the input_text target_text to predict.
            step_size (int): Sliding window step to pass over the input_text.
            batch_size (int): Number of stories to iterate over in parallel.
            dummy (bool): If true then only yield the first 10000 examples.
            **kwargs: Pass to parent.
        """
        self.data_url = data_url
        self.data_download_num_bytes = data_download_num_bytes
        self.data_download_checksum = data_download_checksum
        self.input_size = input_size
        self.target_size = target_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.dummy = dummy
        self.add_negative_examples = add_negative_examples

        super(WritingPromptsInterleavedHfDatasetConfig, self).__init__(**kwargs)


class WritingPromptsInterleavedDataset(datasets.GeneratorBasedBuilder):
    """The WritingPrompts dataset is over 300K short stories collected from the reddit forum /r/WritingPrompts/ .
        Each story is a creative writing exercise following a prompt.
    """

    BUILDER_CONFIG_CLASS = WritingPromptsInterleavedHfDatasetConfig
    BUILDER_CONFIGS = [
        WritingPromptsInterleavedHfDatasetConfig(name="writingprompts_dummy",
                                                 description="Writing Prompts dummy for testng purposes.",
                                                 data_url=_URL,
                                                 data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                                 data_download_checksum=_DOWNLOAD_CHECKSUM,
                                                 dummy=True,
                                                 version=_VERSION),
        WritingPromptsInterleavedHfDatasetConfig(name="writingprompts_dummy_4_label_1_step_4",
                                                 description="Writing Prompts dummy for testng purposes.",
                                                 data_url=_URL,
                                                 input_size=4,
                                                 target_size=1,
                                                 step_size=4,
                                                 data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                                 data_download_checksum=_DOWNLOAD_CHECKSUM,
                                                 dummy=True,
                                                 version=_VERSION),
        WritingPromptsInterleavedHfDatasetConfig(name="writingprompts_dummy_4_label_4_step_4",
                                                 description="Writing Prompts dummy for testng purposes.",
                                                 data_url=_URL,
                                                 input_size=4,
                                                 target_size=4,
                                                 step_size=4,
                                                 data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                                 data_download_checksum=_DOWNLOAD_CHECKSUM,
                                                 dummy=True,
                                                 version=_VERSION),
        WritingPromptsInterleavedHfDatasetConfig(name="writingprompts_dummy_4_label_4_step_4_neg",
                                                 description="Writing Prompts dummy for testng purposes.",
                                                 data_url=_URL,
                                                 input_size=4,
                                                 target_size=4,
                                                 step_size=4,
                                                 data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                                 data_download_checksum=_DOWNLOAD_CHECKSUM,
                                                 dummy=True,
                                                 add_negative_examples=True,
                                                 version=_VERSION),
        WritingPromptsInterleavedHfDatasetConfig(name="writingprompts_context_1_label_1_step_1",
                                                 description="Writing Prompts with one sentence of context, "
                                                             "labels and a one sentence step.",
                                                 data_url=_URL,
                                                 data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                                 data_download_checksum=_DOWNLOAD_CHECKSUM,
                                                 version=_VERSION),
        WritingPromptsInterleavedHfDatasetConfig(name="writingprompts_context_4_label_1_step_4",
                                                 description="Writing Prompts with 4 sentence steps.",
                                                 input_size=4,
                                                 target_size=1,
                                                 step_size=4,
                                                 data_url=_URL,
                                                 data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                                 data_download_checksum=_DOWNLOAD_CHECKSUM,
                                                 version=_VERSION),
        WritingPromptsInterleavedHfDatasetConfig(name="writingprompts_context_4_label_4_step_4",
                                                 description="Writing Prompts with 4 sentence steps.",
                                                 input_size=4,
                                                 target_size=4,
                                                 step_size=4,
                                                 data_url=_URL,
                                                 data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                                 data_download_checksum=_DOWNLOAD_CHECKSUM,
                                                 version=_VERSION)

    ]

    def _info(self):
        ''' Disabled as there is a problem with the checksums not matching.
        download_size = None
        if self.config.data_download_num_bytes is not None:
            download_size = self.config.data_download_num_bytes

        download_checksums = None
        if self.config.data_download_checksum is not None:
            download_checksums = { self.config.data_url: self.config.data_download_checksum}
        '''

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),  # The unique id for the sliding window.
                    "episode_id": datasets.Value("string"),  # The original id.
                    "episode_seq_num": datasets.Value("int32"),  # Unique sequence number for the episode.
                    "title": datasets.Value("string"),  # The title of the work, or for WP the prompt.
                    "text": datasets.Value("string"),  # The context input_text field.
                    "label": datasets.Value("string"),  # The input_text to predict.
                    "negative_labels": [datasets.Value("string")],  # The input_text to predict.
                    "episode_done": datasets.Value("bool"),  # True for the last passage in an episode.
                    "episode_begun": datasets.Value("bool")  # True for the first passage in an episode.

                }
            ),
            # download_size=download_size,
            # download_checksums=download_checksums,
            supervised_keys=SupervisedKeysData(input="text", output="label"),
            version=_VERSION,
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager):
        """Returns splits from train,valid,test.jsonl """

        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "WritingPrompts")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "test.jsonl"), "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "valid.jsonl"),
                    "split": "valid",
                },
            ),
        ]


    def _generate_examples(self, filepath, split):
        """ Yields an example for each story split by stories.
            The prompt is the title but also prepended to the main input_text.
        """

        with jsonlines.open(filepath, mode='r') as reader:
            for example in interleave_examples(reader, self.config.batch_size, self.config.input_size,
                                               self.config.target_size,
                                               self.config.step_size,
                                               dummy=self.config.dummy,
                                               contractions=True,
                                               add_negative_examples=self.config.add_negative_examples):
                yield example['id'], example

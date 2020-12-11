from typing import Optional

import datasets
from datasets.info import SupervisedKeysData
from jsonlines import jsonlines

from story_fragments.data.hf_interleaving_utils import interleave_examples

_CITATION = ""

_DESCRIPTION = """\
 English language plots taken from the English Wikipedia from films, books, plays and other narrative forms. The dataset
 has 132,358 plots in total.
"""

_VERSION = datasets.Version("1.0.0")

_URL = "https://drive.google.com/uc?export=download&id=1PUsmqVzB8SRIRFkBrHAJCL3guojnCjh7"

_HOMEPAGE = "https://github.com/markriedl/WikiPlots"

_DOWNLOAD_NUM_BYTES = 109300457
_DOWNLOAD_CHECKSUM = "7fe76225dcff4ff53830f7272d298a9c2f57e091f76411c652db7b2fed04ed78"


class WikiPlotsInterleavedHfDatasetConfig(datasets.BuilderConfig):

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
        self.add_negative_examples =  add_negative_examples

        super(WikiPlotsInterleavedHfDatasetConfig, self).__init__(**kwargs)


class WritingPromptsInterleavedDataset(datasets.GeneratorBasedBuilder):
    """The WritingPrompts dataset is over 300K short stories collected from the reddit forum /r/WritingPrompts/ .
        Each story is a creative writing exercise following a prompt.
    """

    BUILDER_CONFIG_CLASS = WikiPlotsInterleavedHfDatasetConfig
    BUILDER_CONFIGS = [
        WikiPlotsInterleavedHfDatasetConfig(name="wikiplots_dummy",
                                            description="Wikiplots dummy smaller dataset.",
                                            data_url=_URL,
                                            data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                            data_download_checksum=_DOWNLOAD_CHECKSUM,
                                            version=_VERSION,
                                            dummy=True),
        WikiPlotsInterleavedHfDatasetConfig(name="wikiplots_dummy_4_label_1_step_4",
                                            description="Wikiplots dummy smaller dataset.",
                                            data_url=_URL,
                                            input_size=4,
                                            target_size=1,
                                            step_size=4,
                                            data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                            data_download_checksum=_DOWNLOAD_CHECKSUM,
                                            version=_VERSION,
                                            dummy=True),
        WikiPlotsInterleavedHfDatasetConfig(name="wikiplots_dummy_4_label_4_step_4",
                                            description="Wikiplots dummy smaller dataset.",
                                            data_url=_URL,
                                            input_size=4,
                                            target_size=1,
                                            step_size=4,
                                            data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                            data_download_checksum=_DOWNLOAD_CHECKSUM,
                                            version=_VERSION,
                                            dummy=True),
        WikiPlotsInterleavedHfDatasetConfig(name="wikiplots_dummy_4_label_4_step_4_neg",
                                            description="Wikiplots dummy smaller dataset.",
                                            data_url=_URL,
                                            input_size=4,
                                            target_size=1,
                                            step_size=4,
                                            data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                            data_download_checksum=_DOWNLOAD_CHECKSUM,
                                            version=_VERSION,
                                            dummy=True,
                                            add_negative_examples=True),
        WikiPlotsInterleavedHfDatasetConfig(name="wikiplots_context_1_label_1_step_1",
                                            description="Wikiplots with one sentence of context, "
                                                        "labels and a one sentence step.",
                                            data_url=_URL,
                                            data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                            data_download_checksum=_DOWNLOAD_CHECKSUM,
                                            version=_VERSION),
        WikiPlotsInterleavedHfDatasetConfig(name="wikiplots_context_4_label_1_step_4",
                                            description="Wikiplots with one sentence of context, "
                                                        "labels and a one sentence step.",
                                            input_size=4,
                                            target_size=1,
                                            step_size=4,
                                            data_url=_URL,
                                            data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                            data_download_checksum=_DOWNLOAD_CHECKSUM,
                                            version=_VERSION),
        WikiPlotsInterleavedHfDatasetConfig(name="wikiplots_context_4_label_4_step_4",
                                            description="Wikiplots with one sentence of context, "
                                                        "labels and a one sentence step.",
                                            input_size=4,
                                            target_size=4,
                                            step_size=4,
                                            data_url=_URL,
                                            data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                            data_download_checksum=_DOWNLOAD_CHECKSUM,
                                            version=_VERSION),
        WikiPlotsInterleavedHfDatasetConfig(name="wikiplots_context_4_label_4_step_4_neg",
                                            description="Wikiplots with one sentence of context, "
                                                        "labels and a one sentence step.",
                                            input_size=4,
                                            target_size=4,
                                            step_size=4,
                                            add_negative_examples=True,
                                            data_url=_URL,
                                            data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                            data_download_checksum=_DOWNLOAD_CHECKSUM,
                                            version=_VERSION)

    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),  # The unique id for the sliding window.
                    "episode_id": datasets.Value("string"),  # The original id.
                    "episode_seq_num": datasets.Value("int32"),  # Unique sequence number for the episode.
                    "title": datasets.Value("string"),  # The title of the work.
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
        """Train only as will use datasets functionality to split dynamically."""

        dl_file = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": dl_file,
                    "split": "train",
                },
            )
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
                                               add_negative_examples=self.config.add_negative_examples):
                yield example['id'], example

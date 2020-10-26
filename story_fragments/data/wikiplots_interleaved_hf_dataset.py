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
                 context_size: int = 1,
                 label_size: int = 1,
                 step_size: int = 1,
                 batch_size: int = 100,
                 dummy: bool = False,
                 **kwargs):
        """ Generic config for reading a dataset in a interleaved or round robin fashion.

        Args:
            data_url (str): The url for the compressed jsonl file.
            data_download_num_bytes (int): Number of bytes of the datafile.
            data_download_checksum (str): SHA-256 checksum for the data file.
            context_size (int): Size in sentences of the context text to condition on.
            label_size (int): Size in sentences of the text label to predict.
            step_size (int): Sliding window step to pass over the text.
            batch_size (int): Number of stories to iterate over in parallel.  
            **kwargs: Pass to parent.
        """
        self.data_url = data_url
        self.data_download_num_bytes = data_download_num_bytes
        self.data_download_checksum = data_download_checksum
        self.context_size = context_size
        self.label_size = label_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.dummy = dummy

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
                                            dummy=False),
        WikiPlotsInterleavedHfDatasetConfig(name="wikiplots_context_1_label_1_step_1",
                                            description="Wikiplots with one sentence of context, "
                                                        "labels and a one sentence step.",
                                            data_url=_URL,
                                            data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                            data_download_checksum=_DOWNLOAD_CHECKSUM,
                                            version=_VERSION),
        WikiPlotsInterleavedHfDatasetConfig(name="wikiplots_context_3_label_3_step_3",
                                            description="Wikiplots with one sentence of context, "
                                                        "labels and a one sentence step.",
                                            context_size=3,
                                            label_size=3,
                                            step_size=3,
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
                    "text": datasets.Value("string"),  # The context text field.
                    "label": datasets.Value("string"),  # The text to predict.
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
            The prompt is the title but also prepended to the main text.
        """

        with jsonlines.open(filepath, mode='r') as reader:
            for example in interleave_examples(reader, self.config.batch_size, self.config.context_size,
                                               self.config.label_size,
                                               self.config.step_size,
                                               dummy=self.config.dummy):
                yield example['id'], example

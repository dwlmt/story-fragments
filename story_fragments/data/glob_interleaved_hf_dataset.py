# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
""" Wrapper for corpora like the BookCorpus ."""

from __future__ import absolute_import, division, print_function

import glob
import os
import pathlib
from random import Random
import datasets
from datasets.info import SupervisedKeysData
from story_fragments.data.hf_interleaving_utils import interleave_examples

_DESCRIPTION = """\
Wrapper for multiple interleaving versions of a glob corpus.
"""

_VERSION = datasets.Version("1.0.0")

_CITATION = """\

"""
_PROJECT_URL = ""

_BOOK_CORPUS_URL = "https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz"
_BOOK_CORPUS_GLOB_PATH = "**/*.epub.txt"

_SCHMOOP_CORPUS_URL = "https://drive.google.com/uc?export=download&id=1hhDWhVr1PhFfPj63h6MYACtfBttK_Usl"
_SCHMOOP_CORPUS_GLOB_PATH = "**/*.txt.utf8"

_MOVIE_CORPUS_URL = "https://drive.google.com/uc?export=download&id=16DBMpLY-w5ZF0yph-D3lhRjS_Cgwj-vZ"
_MOVIE_CORPUS_GLOB_PATH = "**/scripts/parsed/full/*.txt"


class GlobInterleavedHfDatasetConfig(datasets.BuilderConfig):

    def __init__(self,
                 data_url: str,
                 glob_path: str,
                 input_size: int = 1,
                 target_size: int = 1,
                 step_size: int = 1,
                 batch_size: int = 60,
                 dummy: bool = False,
                 shuffle: bool = True,
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
        self.glob_path = glob_path
        self.input_size = input_size
        self.target_size = target_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.dummy = dummy
        self.shuffle = True

        super(GlobInterleavedHfDatasetConfig, self).__init__(**kwargs)


class GlobCorpusOpen(datasets.GeneratorBasedBuilder):
    """Wrapper for a GLOB text corpus. """

    BUILDER_CONFIG_CLASS = GlobInterleavedHfDatasetConfig
    BUILDER_CONFIGS = [
        GlobInterleavedHfDatasetConfig(name="bookcorpus_dummy_4_label_1_step_4",
                                       description="Writing Prompts dummy for testng purposes.",
                                       data_url=_BOOK_CORPUS_URL,
                                       glob_path=_BOOK_CORPUS_GLOB_PATH,
                                       input_size=4,
                                       target_size=1,
                                       step_size=4,
                                       dummy=True,
                                       version=_VERSION),
        GlobInterleavedHfDatasetConfig(name="bookcorpus_context_4_label_1_step_4",
                                       description="Writing Prompts with 4 sentence steps.",
                                       input_size=4,
                                       target_size=1,
                                       step_size=4,
                                       data_url=_BOOK_CORPUS_URL,
                                       glob_path=_BOOK_CORPUS_GLOB_PATH,
                                       version=_VERSION),
        GlobInterleavedHfDatasetConfig(name="schmoop_dummy_4_label_1_step_4",
                                       description="Schmoop dummy for testing purposes.",
                                       data_url=_BOOK_CORPUS_URL,
                                       glob_path=_BOOK_CORPUS_GLOB_PATH,
                                       input_size=4,
                                       target_size=1,
                                       step_size=4,
                                       shuffle=False,
                                       dummy=True,
                                       version=_VERSION),
        GlobInterleavedHfDatasetConfig(name="schmoop_context_4_label_1_step_4",
                                       description="Schmoop with 4 sentence steps.",
                                       input_size=4,
                                       target_size=1,
                                       step_size=4,
                                       shuffle=False,
                                       data_url=_BOOK_CORPUS_URL,
                                       glob_path=_BOOK_CORPUS_GLOB_PATH,
                                       version=_VERSION),
        GlobInterleavedHfDatasetConfig(name="moviecorpus_dummy_4_label_1_step_4",
                                       description="Movie script dummy for testing purposes.",
                                       data_url=_MOVIE_CORPUS_URL,
                                       glob_path=_MOVIE_CORPUS_GLOB_PATH,
                                       input_size=4,
                                       target_size=1,
                                       step_size=4,
                                       dummy=True,
                                       version=_VERSION),
        GlobInterleavedHfDatasetConfig(name="moviecorpus_context_4_label_1_step_4",
                                       description="Movie script with 4 sentence steps.",
                                       input_size=4,
                                       target_size=1,
                                       step_size=4,
                                       data_url=_MOVIE_CORPUS_URL,
                                       glob_path=_MOVIE_CORPUS_GLOB_PATH,
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
                    "title": datasets.Value("string"),  # The title of the work, or for WP the prompt.
                    "text": datasets.Value("string"),  # The context input_text field.
                    "label": datasets.Value("string"),  # The input_text to predict.
                    "episode_done": datasets.Value("bool"),  # True for the last passage in an episode.
                    "episode_begun": datasets.Value("bool")  # True for the first passage in an episode.

                }
            ),
            # download_size=download_size,
            # download_checksums=download_checksums,
            supervised_keys=SupervisedKeysData(input="text", output="label"),
            version=_VERSION,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        arch_path = dl_manager.download_and_extract(self.config.data_url)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"directory": arch_path}),
        ]

    def _generate_examples(self, directory):
        glob_target = os.path.join(directory, self.config.glob_path)
        book_files = glob.glob(glob_target, recursive=True)

        if self.config.shuffle:
            # Fix using the default Allennlp random seed.
            Random(13370).shuffle(book_files)
        else:
            book_files = sorted(book_files)

        def _reader(book_files):
            _id = 0
            for book_file_path in book_files:
                path = pathlib.PurePath(book_file_path)
                with open(book_file_path, mode="r", encoding="utf-8") as f:
                    glob_dict = {"title": str(path.name), "text": f.read(), "id": _id}
                    yield glob_dict
                    _id += 1

        for example in interleave_examples(_reader(book_files=book_files), self.config.batch_size,
                                           self.config.input_size,
                                           self.config.target_size,
                                           self.config.step_size,
                                           dummy=self.config.dummy):
            yield example['id'], example

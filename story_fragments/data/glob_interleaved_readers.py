import os
from random import random, randint
from typing import Dict, Iterable

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from datasets import load_dataset


class GlobCorpusInterleavedReader(DatasetReader):
    ''' Interleaved version of the books corpus.

    '''

    def __init__(self,
                 generator_model_name="facebook/bart-base",
                 generator_max_length: int = 128,
                 encoder_model_name="facebook/dpr-question_encoder-multiset-base",
                 encoder_max_length: int = 256,
                 add_special_tokens: bool = True,
                 train_split: int = 80,
                 validation_split: int = 10,
                 test_split: int = 10,
                 manual_shards: int = 10,
                 lazy: bool = True):
        """

        Args:
            generator_model_name (str): Name of the model used for the generator tokenizer.
            generator_max_length (int): Max length.
            encoder_model_name (str): Name of the model used for the encoder (question) tokenizer.
            encoder_max_length (int): Max length.
            add_special_tokens (bool): Whether to add the special BERT tokens.
            train_split (int): % training set split.
            validation_split (int): % validation split.
            test_split (int): % test split.
            **kwargs:
        """
        super().__init__(lazy=lazy)
        self.generator_tokenizer = PretrainedTransformerTokenizer(model_name=generator_model_name,
                                                                  max_length=generator_max_length,
                                                                  add_special_tokens=add_special_tokens,
                                                                  )
        self.generator_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name=generator_model_name, max_length=generator_max_length,

                                                   )}

        self.encoder_tokenizer = PretrainedTransformerTokenizer(model_name=encoder_model_name,
                                                                max_length=encoder_max_length,
                                                                add_special_tokens=add_special_tokens,
                                                                )

        self.encoder_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name=encoder_model_name, max_length=encoder_max_length,
                                                   )}

        assert (train_split + validation_split + test_split) == 100

        self.train_split = train_split
        self.validation_split = validation_split
        self.test_split = test_split

        self.manual_shards = manual_shards

    def text_to_instance(self, example: Dict) -> Instance:
        fields = {}

        fields["metadata"] = MetadataField(example)

        tokens = self.encoder_tokenizer.tokenize(example['text'])

        text_field = TextField(tokens, self.encoder_indexers)
        fields['text'] = text_field

        if "label" in example:
            target_tokens = self.generator_tokenizer.tokenize(example['label'])

            fields['labels'] = TextField(target_tokens, self.generator_indexers)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        ''' Read and process the instances.

        Args:
            file_path (str): Because this wraps a Huggingface dataset the path is the Huggingface config name. Separated by the split nameas 'config/split'.

        Returns: Instance

        '''
        split_arr = file_path.split('/')
        config = split_arr[0]
        split = split_arr[1]

        if len(split_arr) == 4:
            shard_index = int(split_arr[2])
            world_size = int(split_arr[3])
        else:
            shard_index = None
            world_size = None

        if split == "train":
            dataset = load_dataset(f"{os.path.dirname(__file__)}/glob_interleaved_hf_dataset.py", name=config,
                                   split=f'train[:{self.train_split}%]')
        elif split == "validation":
            dataset = load_dataset(f"{os.path.dirname(__file__)}/glob_interleaved_hf_dataset.py", name=config,
                                   split=f'train[{self.train_split}%:{self.train_split + self.validation_split}%]')
        else:
            dataset = load_dataset(f"{os.path.dirname(__file__)}/glob_interleaved_hf_dataset.py", name=config,
                                   split=f'train[-{self.test_split}%:]')

        print(dataset)

        if shard_index is not None:
            dataset = dataset.shard(world_size, shard_index, contiguous=True)

        if self.manual_shards > 1:
            total_num_examples = len(dataset)
            shard_size = int(total_num_examples / self.manual_shards)
            shard = randint(0, self.manual_shards - 1)
            start = shard * shard_size
            finish = min(start + shard_size, total_num_examples - 1)
            index_range = [r for r in range(start, finish+1)]
            dataset = dataset.select(index_range)


        for example in dataset:
            yield self.text_to_instance(example)


@DatasetReader.register('bookcorpus-interleaved')
class BookCorpusInterleavedReader(GlobCorpusInterleavedReader):
    ''' Interleaved version of the books corpus.

    '''

    def __init__(self,
                 generator_model_name="facebook/bart-base",
                 generator_max_length: int = 128,
                 encoder_model_name="facebook/dpr-question_encoder-multiset-base",
                 encoder_max_length: int = 256,
                 add_special_tokens: bool = True,
                 train_split: int = 80,
                 validation_split: int = 10,
                 test_split: int = 10,
                 **kwargs):

        super(BookCorpusInterleavedReader, self).__init__(generator_model_name=generator_model_name,
                                                          generator_max_length=generator_max_length,
                                                          encoder_model_name=encoder_model_name,
                                                          encoder_max_length=encoder_max_length,
                                                          add_special_tokens=add_special_tokens,
                                                          train_split=train_split,
                                                          validation_split=validation_split,
                                                          test_split=test_split,
                                                          **kwargs
                                                          )


@DatasetReader.register('schmoop-interleaved')
class SchmoopCorpusInterleavedReader(GlobCorpusInterleavedReader):
    ''' Interleaved version of the books corpus.

    '''

    def __init__(self,
                 generator_model_name="facebook/bart-base",
                 generator_max_length: int = 128,
                 encoder_model_name="facebook/dpr-question_encoder-multiset-base",
                 encoder_max_length: int = 256,
                 add_special_tokens: bool = True,
                 train_split: int = 80,
                 validation_split: int = 10,
                 test_split: int = 10,
                 **kwargs):

        super(SchmoopCorpusInterleavedReader, self).__init__(generator_model_name=generator_model_name,
                                                          generator_max_length=generator_max_length,
                                                          encoder_model_name=encoder_model_name,
                                                          encoder_max_length=encoder_max_length,
                                                          add_special_tokens=add_special_tokens,
                                                          train_split=train_split,
                                                          validation_split=validation_split,
                                                          test_split=test_split,
                                                          **kwargs
                                                          )

@DatasetReader.register('moviecorpus-interleaved')
class MovieCorpusInterleavedReader(GlobCorpusInterleavedReader):
    ''' Interleaved version of the books corpus.

    '''

    def __init__(self,
                 generator_model_name="facebook/bart-base",
                 generator_max_length: int = 128,
                 encoder_model_name="facebook/dpr-question_encoder-multiset-base",
                 encoder_max_length: int = 256,
                 add_special_tokens: bool = True,
                 train_split: int = 80,
                 validation_split: int = 10,
                 test_split: int = 10,
                 **kwargs):

        super(MovieCorpusInterleavedReader, self).__init__(generator_model_name=generator_model_name,
                                                          generator_max_length=generator_max_length,
                                                          encoder_model_name=encoder_model_name,
                                                          encoder_max_length=encoder_max_length,
                                                          add_special_tokens=add_special_tokens,
                                                          train_split=train_split,
                                                          validation_split=validation_split,
                                                          test_split=test_split,
                                                          **kwargs
                                                          )


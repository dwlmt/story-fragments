import logging
import os
import random
from typing import Dict, Iterable

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from datasets import load_dataset

logger = logging.getLogger(__name__)


@DatasetReader.register('writingprompts-interleaved')
class WritingPromptsInterleavedReader(DatasetReader):
    ''' Interleaved version of WritingPrompts.

    '''

    def __init__(self,
                 generator_model_name="facebook/bart-base",
                 generator_max_length: int = 128,
                 encoder_model_name="facebook/dpr-question_encoder-multiset-base",
                 encoder_max_length: int = 256,
                 add_special_tokens: bool = True,
                 search_negative_labels: bool = False,
                 k_nearest: int = 5,
                 manual_shards: int = 1,
                 max_instances: int = None,
                 lazy: bool = True):
        """
        Args:
            generator_model_name (str): Name of the model used for the generator tokenizer.
            generator_max_length (int): Max length.
            encoder_model_name (str): Name of the model used for the encoder (question) tokenizer.
            encoder_max_length (int): Max length.
            add_special_tokens (bool): Whether to add the special tokens.
            **kwargs: Additional args.
        """
        super().__init__(lazy=lazy, max_instances=max_instances)

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

        self.search_negative_labels = search_negative_labels
        self.k_nearest = k_nearest

        self.manual_shards = manual_shards
        self.seen = {}

    def text_to_instance(self, example: Dict) -> Instance:
        fields = {}

        fields["metadata"] = MetadataField(example)
        # logger.info(f"Example: {example}")

        tokens = self.encoder_tokenizer.tokenize(example['text'])

        text_field = TextField(tokens, self.encoder_indexers)
        fields['text'] = text_field

        if "label" in example:
            target_tokens = self.generator_tokenizer.tokenize(example['label'])

            fields['labels'] = TextField(target_tokens, self.generator_indexers)

        if "negative_labels" in example:

            negative_labels = example["negative_labels"]

            if len(negative_labels) > 1:
                negative_label = random.choice(negative_labels)
            elif len(negative_labels) > 0:
                negative_label = negative_labels[0]
            else:
                negative_label = "<BLANK>"

            if negative_label is not None:
                negative_label_tokens = self.generator_tokenizer.tokenize(negative_label)
                fields['negative_labels'] = TextField(negative_label_tokens, self.generator_indexers)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        ''' Read and process the instances.

        Args:
            file_path (str): Because this wraps a Huggingface dataset the path is the Huggingface config name. Separated by the split nameas 'config/split'.

        Returns: Instance

        '''

        if file_path not in self.seen:
            self.seen[file_path] = 0
        else:
            self.seen[file_path] += 1

        split_arr = file_path.split('/')
        config = split_arr[0]
        split = split_arr[1]

        dataset = load_dataset(f"{os.path.dirname(__file__)}/writingprompts_interleaved_hf_dataset.py", name=config,
                               split=split)

        if self.manual_shards > 1 and not self.search_negative_labels:
            dataset = dataset.shard(self.manual_shards, random.randrange(0, self.manual_shards), contiguous=True)

        if self.search_negative_labels:
            dataset.add_elasticsearch_index("label", host="localhost", port="9200")

        for i, example in enumerate(dataset):

            if self.search_negative_labels and self.seen[file_path] > 0:
                try:
                    label = example["label"]
                    neg_examples = dataset.get_nearest_examples("label", label, k=1 + self.k_nearest).examples['label'][
                                   1:]
                    example["negative_labels"].extend(neg_examples)
                except Exception as e:
                    print(f"Failed retrieval: {e}")

            yield self.text_to_instance(example)

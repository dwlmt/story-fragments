import os
from typing import Dict, Iterable

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from datasets import load_dataset


@DatasetReader.register('wikiplots-interleaved')
class WikiplotsInterleavedReader(DatasetReader):
    ''' Interleaved version of Wikiplots

    Wikiplots has a single soure file and so is dynamically split.

    '''

    def __init__(self,
                 transformer_model_name="facebook/bart-base",
                 max_length: int = 128,
                 add_special_tokens: bool = True,
                 train_split: int = 80,
                 validation_split: int = 10,
                 test_split: int = 10,
                 **kwargs):
        """

        Args:
              transformer_model_name (str): Name of the transformers library tokenizer.
            max_length (int): Max length
            add_special_tokens (bool): Whether to add the special BERT tokens.
            train_split (int): % training set split.
            validation_split (int): % validation split.
            test_split (int): % test split.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.tokenizer = PretrainedTransformerTokenizer(model_name=transformer_model_name, max_length=max_length,
                                                        add_special_tokens=add_special_tokens,
                                                        tokenizer_kwargs={"truncation": True})
        self.token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name=transformer_model_name, max_length=max_length)}

        assert (train_split + validation_split + test_split) == 100

        self.train_split = train_split
        self.validation_split = validation_split
        self.test_split = test_split

    def text_to_instance(self, example: Dict) -> Instance:
        fields = {}

        fields["metadata"] = MetadataField(example)

        tokens = self.tokenizer.tokenize(example['text'])

        text_field = TextField(tokens, self.token_indexers)
        fields['text'] = text_field

        if "label" in example:
            target_tokens = self.tokenizer.tokenize(example['label'])
            # This will need to be manipulated.

            fields['labels'] = TextField(target_tokens, self.token_indexers)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        ''' Read and process the instances.

        Args:
            file_path (str): Because this wraps a Huggingface dataset the path is the Huggingface config name. Separated by the split nameas 'config/split'.

        Returns: Instance

        '''
        config, split = file_path.split('/')

        if split == "train":
            dataset = load_dataset(f"{os.path.dirname(__file__)}/wikiplots_interleaved_hf_dataset.py", name=config,
                                   split=f'train[:{self.train_split}%]')
        elif split == "validation":
            dataset = load_dataset(f"{os.path.dirname(__file__)}/wikiplots_interleaved_hf_dataset.py", name=config,
                                   split=f'train[{self.train_split}%:{self.train_split + self.validation_split}%]')
        else:
            dataset = load_dataset(f"{os.path.dirname(__file__)}/wikiplots_interleaved_hf_dataset.py", name=config,
                                   split=f'train[-{self.test_split}%:]')

        for example in dataset:
            yield self.text_to_instance(example)

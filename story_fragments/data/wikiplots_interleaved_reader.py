import os
from typing import Dict, Iterable

from allennlp.data import DatasetReader, Tokenizer, TokenIndexer, Instance
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
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 train_split: int = 80,
                 valid_split: int = 10,
                 test_split: int = 10,
                 **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name="facebook/bart-base")
        self.token_indexers = token_indexers or {
            "tokens": PretrainedTransformerIndexer(model_name="facebook/bart-base")}

        self.train_split = train_split
        self.valid_split = valid_split
        self.test_split = test_split

    def text_to_instance(self, example: Dict) -> Instance:
        fields = {}

        fields["metadata"] = MetadataField(example)

        tokens = self.tokenizer.tokenize(example['text'])

        text_field = TextField(tokens, self.token_indexers)
        fields['text'] = text_field
        if "label" in example:
            label_tokens = self.tokenizer.tokenize(example['label'])
            fields['label'] = TextField(label_tokens, self.token_indexers)

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
                                   split=f'train[{self.train_split}%:{self.train_split + self.valid_split}%]')
        else:
            dataset = load_dataset(f"{os.path.dirname(__file__)}/wikiplots_interleaved_hf_dataset.py", name=config,
                                   split=f'train[-{self.test_split}%:]')

        for example in dataset:
            yield self.text_to_instance(example)

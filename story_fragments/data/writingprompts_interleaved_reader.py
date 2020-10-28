import os
from typing import Dict, Iterable

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from datasets import load_dataset


@DatasetReader.register('writingprompts-interleaved')
class WritingPromptsInterleavedReader(DatasetReader):
    ''' Interleaved version of WritingPrompts.

    '''

    def __init__(self,
                 transformer_model_name="facebook/bart-base",
                 max_length: int = 128,
                 add_special_tokens: bool = True,
                 **kwargs):
        """
        Args:
            transformer_model_name (str): Name of the transformers library tokenizer.
            max_length (int): Max length
            add_special_tokens (bool): Whether to add the special BERT tokens.
            **kwargs: Additional args.
        """
        super().__init__(**kwargs)
        self.tokenizer = PretrainedTransformerTokenizer(model_name=transformer_model_name, max_length=max_length,
                                                        add_special_tokens=add_special_tokens,
                                                        tokenizer_kwargs={"truncation": True})
        self.token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name=transformer_model_name, max_length=max_length)}

    def text_to_instance(self, example: Dict) -> Instance:
        fields = {}

        fields["metadata"] = MetadataField(example)

        tokens = self.tokenizer.tokenize(example['text'])

        text_field = TextField(tokens, self.token_indexers)
        fields['text'] = text_field

        if "label" in example:
            label_tokens = self.tokenizer.tokenize(example['label'])

            fields['labels'] = TextField(label_tokens, self.token_indexers)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        ''' Read and process the instances.

        Args:
            file_path (str): Because this wraps a Huggingface dataset the path is the Huggingface config name. Separated by the split nameas 'config/split'.

        Returns: Instance

        '''
        config, split = file_path.split('/')

        dataset = load_dataset(f"{os.path.dirname(__file__)}/writingprompts_interleaved_hf_dataset.py", name=config,
                               split=split)

        for example in dataset:
            yield self.text_to_instance(example)

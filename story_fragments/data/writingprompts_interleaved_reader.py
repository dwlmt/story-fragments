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
                 generator_model_name="facebook/bart-base",
                 generator_max_length: int = 128,
                 encoder_model_name="facebook/dpr-question_encoder-multiset-base",
                 encoder_max_length: int = 256,
                 add_special_tokens: bool = True,
                 **kwargs):
        """
        Args:
            generator_model_name (str): Name of the model used for the generator tokenizer.
            generator_max_length (int): Max length.
            encoder_model_name (str): Name of the model used for the encoder (question) tokenizer.
            encoder_max_length (int): Max length.
            add_special_tokens (bool): Whether to add the special tokens.
            **kwargs: Additional args.
        """
        super().__init__(**kwargs)

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
        config, split = file_path.split('/')

        dataset = load_dataset(f"{os.path.dirname(__file__)}/writingprompts_interleaved_hf_dataset.py", name=config,
                               split=split)

        for example in dataset:
            yield self.text_to_instance(example)

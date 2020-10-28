from typing import Dict, Any, List

import torch

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from transformers import AutoModel


@Model.register('toy-lm-story')
class TransformerToyModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 transformer_model_name: str = "facebook/bart-base"):
        super().__init__(vocab)

        self.transformer_model_name = transformer_model_name
        self.transformer = AutoModel.from_pretrained(self.transformer_model_name)

    # Note that the signature of forward() needs to match that of field names
    def forward(self,
                text: TextFieldTensors,
                label: TextFieldTensors = None,
                metadata: List[Dict[str, Any]] = None,
                dataset: List[str] = None
                ) -> Dict[str, torch.Tensor]:

        print('tokens:', text)
        print('label:', label)
        print('metadata:', metadata)
        print('dataset:', dataset)

        return {}

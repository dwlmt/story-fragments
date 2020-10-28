from typing import Dict, Any, List

import torch

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from transformers import AutoModel, BartForConditionalGeneration


@Model.register('bart-fragments')
class TransformerToyModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 transformer_model_name: str = "facebook/bart-base"):
        super().__init__(vocab)

        self.transformer_model_name = transformer_model_name
        self.transformer = BartForConditionalGeneration.from_pretrained(self.transformer_model_name)

    # Note that the signature of forward() needs to match that of field names
    def forward(self,
                text: TextFieldTensors,
                labels: TextFieldTensors = None,
                decoder_text: TextFieldTensors = None,
                metadata: List[Dict[str, Any]] = None,
                dataset: List[str] = None
                ) -> Dict[str, torch.Tensor]:

        results = {}

        input_ids = text["tokens"]['token_ids']

        if labels is not None:
            labels = labels["tokens"]['token_ids']
            decoder_text = decoder_text["tokens"]['token_ids'] if decoder_text is not None else None

            trans_out = self.transformer(input_ids=input_ids, decoder_input=decoder_text, labels=labels)

            results["loss"] = trans_out[0]

        return results

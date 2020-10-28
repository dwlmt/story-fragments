from typing import Dict, Any, List

import torch

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.training.metrics import Metric, SequenceAccuracy, Perplexity
from transformers import AutoModel, BartForConditionalGeneration


@Model.register('bart-fragments')
class TransformerToyModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 transformer_model_name: str = "facebook/bart-base"):
        super().__init__(vocab)

        self.transformer_model_name = transformer_model_name
        self.transformer = BartForConditionalGeneration.from_pretrained(self.transformer_model_name)

        self.metrics = {}
        self.metrics['accuracy'] = SequenceAccuracy()
        self.metrics['perplexity'] = Perplexity()

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
            label_tokens = labels["tokens"]['token_ids']
            decoder_tokens = decoder_text["tokens"]['token_ids'] if decoder_text is not None else None

            trans_output = self.transformer(input_ids=input_ids, decoder_input=decoder_tokens, labels=label_tokens,
                                            return_dict=True)

            results["loss"] = trans_output["loss"]

            self._update_metrics(trans_output, label_tokens)

        return results

    def _update_metrics(self, trans_output, label_tokens):
        # Only update metrics when not training to improve performance
        if not self.training:
            self.metrics['perplexity'](trans_output["loss"])

            logits = trans_output["logits"]
            logits_permuted = logits.permute(0, 2, 1)
            mask = (label_tokens != 1).bool()
            self.metrics['accuracy'](logits_permuted, label_tokens, mask=mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

        for k, v in metrics.items():

            if isinstance(v, Dict):
                metrics = {**metrics, **v}

        return metrics

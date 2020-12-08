import logging
from typing import Dict, Any, List

import torch

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.training.metrics import Metric, SequenceAccuracy, Perplexity, CategoricalAccuracy
from transformers import AutoModel, BartForConditionalGeneration, AutoTokenizer

PAD_TOKEN = 1

logger = logging.getLogger(__name__)


@Model.register('bart-fragments')
class BartFragmentsModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 transformer_model_name: str = "facebook/bart-base",
                 lm_accuracy_top_k: List[int] = [1, 5, 20]):
        super().__init__(vocab)

        self.transformer = BartForConditionalGeneration.from_pretrained(transformer_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

        self.lm_accuracy_top_k = lm_accuracy_top_k
        self.metrics = {}

        for acc in self.lm_accuracy_top_k:
            self.metrics[f'lm_accuracy_{acc}'] = CategoricalAccuracy()
        self.metrics['lm_perplexity'] = Perplexity()

    # Note that the signature of forward() needs to match that of field names
    def forward(self,
                text: TextFieldTensors,
                labels: TextFieldTensors = None,
                negative_labels: TextFieldTensors = None,
                metadata: List[Dict[str, Any]] = None,
                dataset: List[str] = None,
                num_sequences_to_generate: int = 0,
                ) -> Dict[str, torch.Tensor]:

        results = {}

        input_ids = text["tokens"]['token_ids']
        input_mask = text["tokens"]['mask']

        if labels is not None:
            label_tokens = labels["tokens"]['token_ids']

            trans_output = self.transformer(input_ids=input_ids,
                                            #attention_mask=input_mask,
                                            labels=label_tokens)

            results["loss"] = trans_output[0]

            self._update_metrics(trans_output, label_tokens)

        self._generate_if_required(input_ids, input_mask, num_sequences_to_generate, results)

        return results

    def _generate_if_required(self, input_ids, input_mask, num_sequences_to_generate, results):
        if not self.training and num_sequences_to_generate > 0:
            with torch.no_grad():
                # TODO: Make the generation parameters configurable.
                generated_sequences = self.model.generate(input_ids=input_ids,
                                                          attention_mask=input_mask,
                                                          num_return_sequences=num_sequences_to_generate,
                                                          do_sample=True,
                                                          max_length=64,
                                                          top_p=0.9)
                results["generated_sequences"] = generated_sequences

    def make_output_human_readable( self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "generated_sequences" in output_dict:
            output_dict["generated_sequences"] = self.tokenizer.decode(output_dict["generated_sequences"],
                                                                       skip_special_tokens=True)
        return output_dict

    def _update_metrics(self, model_output, label_tokens):
        with torch.no_grad():
            # Only update metrics when not training to improve performance
            if not self.training:
                self.metrics['lm_perplexity'](torch.mean(model_output[0]))

                logits = model_output[1]

                mask = (label_tokens != PAD_TOKEN)

                for acc in self.lm_accuracy_top_k:
                    #print(logits.size(), label_tokens.size())
                    self.metrics[f'lm_accuracy_{acc}'](logits, label_tokens, mask=mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

        for k, v in metrics.items():

            if isinstance(v, Dict):
                metrics = {**metrics, **v}

        return metrics

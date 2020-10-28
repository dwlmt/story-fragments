from typing import Dict, Any, List

import torch

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.training.metrics import Metric, SequenceAccuracy, Perplexity
from transformers import AutoModel, BartForConditionalGeneration, AutoTokenizer

PAD_TOKEN = 1


@Model.register('bart-fragments')
class BartFragmentsModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 transformer_model_name: str = "facebook/bart-base"):
        super().__init__(vocab)

        self.transformer = BartForConditionalGeneration.from_pretrained(transformer_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

        self.metrics = {}
        self.metrics['accuracy'] = SequenceAccuracy()
        self.metrics['perplexity'] = Perplexity()

    # Note that the signature of forward() needs to match that of field names
    def forward(self,
                text: TextFieldTensors,
                labels: TextFieldTensors = None,
                decoder_text: TextFieldTensors = None,
                metadata: List[Dict[str, Any]] = None,
                dataset: List[str] = None,
                num_sequences_to_generate: int = 0,
                ) -> Dict[str, torch.Tensor]:

        results = {}

        input_ids = text["tokens"]['token_ids']
        input_mask = text["tokens"]['mask']

        if labels is not None:
            label_tokens = labels["tokens"]['token_ids']
            decoder_tokens = decoder_text["tokens"]['token_ids'] if decoder_text is not None else None

            trans_output = self.transformer(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            decoder_input=decoder_tokens,
                                            labels=label_tokens,
                                            return_dict=True)

            results["loss"] = trans_output["loss"]

            self._update_metrics(trans_output, label_tokens)

        if not self.training and num_sequences_to_generate > 0:
            with torch.no_grad():
                # TODO: Make the generation parameters configurable.
                generated_sequences = self.transformer.generate(input_ids=input_ids,
                                                                attention_mask=input_mask,
                                                                num_return_sequences=num_sequences_to_generate,
                                                                do_sample=True,
                                                                max_length=64,
                                                                top_p=0.9)
                results["generated_sequences"] = generated_sequences

        return results

    def make_output_human_readable( self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "generated_sequences" in output_dict:
            output_dict["generated_sequences"] = self.tokenizer.decode(output_dict["generated_sequences"],
                                                                       skip_special_tokens=True)
        return output_dict

    def _update_metrics(self, trans_output, label_tokens):
        with torch.no_grad():
            # Only update metrics when not training to improve performance
            if not self.training:
                self.metrics['perplexity'](trans_output["loss"])

                logits = trans_output["logits"]
                logits_soft = logits.softmax(dim=-1)
                logits_permuted = logits_soft.permute(0, 2, 1)
                mask = (label_tokens != PAD_TOKEN).bool()
                self.metrics['accuracy'](logits_permuted, label_tokens, mask=mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

        for k, v in metrics.items():

            if isinstance(v, Dict):
                metrics = {**metrics, **v}

        return metrics

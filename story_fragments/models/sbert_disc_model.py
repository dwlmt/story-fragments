import logging
from typing import Dict, Any, List

import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy
from sentence_transformers import SentenceTransformer, models
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

PAD_TOKEN = 1

logger = logging.getLogger(__name__)


@Model.register('sbert-disc-fragments')
class SbertDiscFragmentsModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 tokenizer_name="bert-base-cased",
                 model_name: str = "bert-base-cased",
                 max_seq_length: int = 256,
                 lm_accuracy_top_k: List[int] = [1, 5, 20]):
        super().__init__(vocab)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        # self.model = SentenceTransformer(model_name)

        self.lm_accuracy_top_k = lm_accuracy_top_k
        self.metrics = {}

        for acc in self.lm_accuracy_top_k:
            self.metrics[f'lm_accuracy_{acc}'] = CategoricalAccuracy()

    # Note that the signature of forward() needs to match that of field names
    def forward(self,
                text: TextFieldTensors,
                labels: TextFieldTensors = None,
                metadata: List[Dict[str, Any]] = None,
                dataset: List[str] = None,
                num_sequences_to_generate: int = 0,
                ) -> Dict[str, torch.Tensor]:

        results = {}

        input_ids = text["tokens"]['token_ids']

        if labels is not None:
            label_ids = labels["tokens"]['token_ids']

            examples = []
            labels = []

            for input in input_ids:
                examples.append(input)

            for label in label_ids:
                labels.append(label)

            examples_tensor = pad_sequence(examples, batch_first=True, padding_value=0.0)
            examples_tensor = pad_sequence(examples, batch_first=True, padding_value=0.0)

            input_tensor, label_tensor = torch.split(examples_tensor, int(examples_tensor.size()[0] / 2), dim=0)
            input_tensor = input_tensor.unsqueeze(dim=1)
            label_tensor = label_tensor.unsqueeze(dim=1)
            examples_tensor = torch.cat((input_tensor, label_tensor), dim=1)
            examples_tensor = examples_tensor.permute(1, 0, 2)

            examples_mask_tensor = (examples_tensor != 0)

            examples_list = []
            for e, m in zip(examples_tensor, examples_mask_tensor):
                example_dict = {"input_ids": e, "attention_mask": m}
                examples_list.append(example_dict)

            # output = self.loss(examples_list, labels=None)
            # print(f"Examples list: {examples_list}")
            output = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in examples_list]

            reps_a = output[0]
            reps_b = torch.cat(output[1:])
            # print(f"Representations: {reps_a}, {reps_b}, {reps_a.size()}, {reps_b.size()}")

            scores = torch.mm(reps_a, reps_b.transpose(0, 1))

            scores[torch.isnan(scores)] = 0.0

            print(f"Dot Product: {scores}")
            labels = torch.tensor(range(len(scores)), dtype=torch.long,
                                  device=scores.device)
            # print(f"Labels: {labels}")

            self.cross_entropy_loss = CrossEntropyLoss()
            loss = self.cross_entropy_loss(scores, labels)
            # print(f"Loss {loss}")

            results["loss"] = loss

        return results

    def _update_metrics(self, model_output, label_tokens):
        with torch.no_grad():
            # Only update metrics when not training to improve performance
            if not self.training:

                logits = model_output[1]

                mask = (label_tokens != PAD_TOKEN)

                for acc in self.lm_accuracy_top_k:
                    # print(logits.size(), label_tokens.size())
                    self.metrics[f'lm_accuracy_{acc}'](logits, label_tokens, mask=mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

        for k, v in metrics.items():

            if isinstance(v, Dict):
                metrics = {**metrics, **v}

        return metrics

import os
from typing import List

import more_itertools
import nltk
import torch
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import wasserstein_distance
from torch import nn

from story_fragments.predictors.utils import input_to_passages

nltk.download('vader_lexicon')


def parse_bool(b):
    return b == "True" or b == "TRUE" or b == "true" or b == "1"


@Predictor.register("rag-fragments-inference")
class RagFragmentsInferencePredictor(Predictor):
    """
        Exports the
    """

    def __init__(
            self, model: Model,
            dataset_reader: DatasetReader
    ) -> None:
        super().__init__(model, dataset_reader)

        self._sentence_batch_size = int(os.getenv("SENTENCE_BATCH_SIZE", default=6))
        self._sentence_label_size = int(os.getenv("SENTENCE_LABEL_SIZE", default=6))
        self._sentence_step_size = int(os.getenv("SENTENCE_STEP_SIZE", default=6))
        self._max_passages = int(os.getenv("MAX_PASSAGES", default=1000000))

        self._add_to_memory = parse_bool(os.getenv("ADD_TO_MEMORY", default="True"))
        self._keep_embeddings = parse_bool(os.getenv("KEEP_EMBEDDINGS", default="True"))

        generator_model_name = str(os.getenv("GENERATOR_MODEL_NAME", default="facebook/bart-base"))
        generator_max_length = int(os.getenv("GENERATOR_MAX_LENGTH", default=128))
        encoder_model_name = str(os.getenv("ENCODER_MODEL_NAME", default="facebook/dpr-question_encoder-multiset-base"))
        encoder_max_length = int(os.getenv("ENCODER_MAX_LENGTH", default=512))
        add_special_tokens = parse_bool(os.getenv("ADD_SPECIAL_TOKENS", default="True"))

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

        self._cosine_similarity = nn.CosineSimilarity()
        self._l2_distance = nn.PairwiseDistance(p=2)
        self._l1_distance = nn.PairwiseDistance(p=1)

        self._vader_analyzer = SentimentIntensityAnalyzer()

        # Hacky flag to stop adding duplicates to memory.
        os.environ['DONT_ADD_TO_MEMORY'] = 'True'

    def predict(self, sentences: List[str] = None, text: str = None, passage: str = None) -> JsonDict:

        return self.predict_json({"sentence": sentences, "text": text, "passage": passage})

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        results = {}

        results["inputs"] = inputs

        passages = input_to_passages(inputs, sentence_batch_size=self._sentence_batch_size,
                                     sentence_label_size=self._sentence_label_size,
                                     sentence_step_size=self._sentence_step_size,
                                     max_passages=self._max_passages)

        results["passages"] = []

        model_outputs_list = []
        for example in passages:

            if "label" not in example:
                continue

            results["passages"].append(example)

            instance = self._json_to_instance(example)

            outputs = self._model.forward_on_instance(instance)

            sentiment = self._vader_analyzer.polarity_scores(example["text"])
            outputs["vader_sentiment"] = sentiment["compound"]

            print(f"Outputs: {outputs}")

            if self._keep_embeddings:
                example["retrieved_doc_embedding"] = outputs["retrieved_doc_embeddings"].tolist()
                example["generator_enc_embedding"] = outputs["generator_enc_embeddings"].tolist()
                example["generator_dec_embedding"] = outputs["generator_dec_embeddings"].tolist()

            model_outputs_list.append(outputs)

            results["passages"].append(example)

            # print(f"question_encoder_last_hidden_state: {outputs['question_encoder_last_hidden_state'].size()}")

            if self._add_to_memory:
                self._model.add_to_memory(example["text"], add_to_memory=self._add_to_memory)

        for i, (first, second) in enumerate(more_itertools.pairwise(model_outputs_list)):

            if first == None or second == None:
                continue

            def vector_distance_metrics(name, x, y):

                res_dict = {}

                wass_dist = wasserstein_distance(x.numpy(), y.numpy())

                if len(x.size()) < 2:
                    x = torch.unsqueeze(x, dim=0)
                    y = torch.unsqueeze(y, dim=0)

                if len(y.size()) < len(x.size()):
                    y = torch.unsqueeze(y, dim=0).expand_as(x)

                # norm = numpy.linalg.norm(x, ord=2)
                # x = x / norm

                # norm = numpy.linalg.norm(y, ord=2)
                # y = y / norm

                # print(f"Normalized: {x}, {y}")

                l1_dist = self._l1_distance(x, y)
                l2_dist = self._l2_distance(x, y)
                cosine_sim = self._cosine_similarity(x, y)
                cosine_dist = 1.0 - cosine_sim
                dot_product = (x * y).sum(-1)

                res_dict[f"{name}_l1_dist"] = l1_dist.item()
                res_dict[f"{name}_l2_dist"] = l2_dist.item()
                res_dict[f"{name}_cosine_sim"] = cosine_sim.item()
                res_dict[f"{name}_dot_product"] = dot_product.item()
                res_dict[f"{name}_wasserstein_dist"] = wass_dist

                return res_dict

            first_doc_emb = torch.tensor(first["retrieved_doc_embeddings"])
            second_doc_emb = torch.tensor(second["retrieved_doc_embeddings"])
            metrics = vector_distance_metrics("retrieved_doc_embedding", first_doc_emb, second_doc_emb)
            results["passages"][i]["prediction_metrics"] = metrics

            first_doc_emb = torch.tensor(first["generator_enc_embeddings"])
            second_doc_emb = torch.tensor(second["generator_enc_embeddings"])
            metrics = vector_distance_metrics("generator_enc_embedding", first_doc_emb, second_doc_emb)
            results["passages"][i]["prediction_metrics"] = {**metrics, **results["passages"][i]["prediction_metrics"]}

            first_doc_emb = torch.tensor(first["generator_dec_embeddings"])
            second_doc_emb = torch.tensor(second["generator_dec_embeddings"])
            metrics = vector_distance_metrics("generator_dec_embedding", first_doc_emb, second_doc_emb)
            results["passages"][i]["prediction_metrics"] = {**metrics, **results["passages"][i]["prediction_metrics"]}

            if "perplexity" in first:
                results["passages"][i]["prediction_metrics"]["perplexity"] = first["perplexity"].item()

            results["passages"][i]["prediction_metrics"]["vader_sentiment"] = first["vader_sentiment"]

        self._model.clear_memory()

        return results

    def _json_to_instance(self, example: JsonDict) -> Instance:

        fields = {}

        fields["metadata"] = MetadataField(example)
        # logger.info(f"Example: {example}")

        tokens = self.encoder_tokenizer.tokenize(example['text'])

        text_field = TextField(tokens, self.encoder_indexers)
        fields['text'] = text_field

        if "label" in example:
            target_tokens = self.generator_tokenizer.tokenize(example['label'])

            fields['labels'] = TextField(target_tokens, self.generator_indexers)

        return Instance(fields)

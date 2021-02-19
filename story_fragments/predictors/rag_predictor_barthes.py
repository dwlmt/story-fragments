import copy
import os
from random import random, choice
from typing import List

import more_itertools
import nltk
import numpy
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from nltk.sentiment import SentimentIntensityAnalyzer
from overrides import overrides
import torch.nn.functional as F
from scipy.signal import find_peaks
from scipy.special import kl_div
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from transformers import AutoTokenizer
from blingfire import text_to_sentences

from story_fragments.predictors.utils import input_to_passages

nltk.download('vader_lexicon')

def parse_bool(b):
    return b == "True" or b == "TRUE" or b == "true" or b == "1"

embeddings_fields = ["generator_dec_embedding","retrieved_doc_embedding","generator_enc_embedding"]


@Predictor.register("rag-fragments-barthes")
class RagFragmentsBarthesPredictor(Predictor):
    """ Runs predictions used the barthes cardinal function method.
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
        self._keep_embeddings = parse_bool(os.getenv("KEEP_EMBEDDINGS", default="False"))

        generator_model_name = str(os.getenv("GENERATOR_MODEL_NAME", default="facebook/bart-base"))
        generator_max_length = int(os.getenv("GENERATOR_MAX_LENGTH", default=256))
        encoder_model_name = str(os.getenv("ENCODER_MODEL_NAME", default="facebook/dpr-question_encoder-multiset-base"))
        encoder_max_length = int(os.getenv("ENCODER_MAX_LENGTH", default=256))
        add_special_tokens = parse_bool(os.getenv("ADD_SPECIAL_TOKENS", default="True"))


        self._peak_distance = int(os.getenv("PEAK_DISTANCE", default=3))
        self._peak_prominence = float(os.getenv("PEAK_PROMINENCE", default=0.10))
        self._peak_threshold = float(os.getenv("PEAK_THRESHOLD", default=0.025))
        self._peak_height = float(os.getenv("PEAK_HEIGHT", default=0.025))

        self._add_retrieved_docs = parse_bool(os.getenv("ADD_RETRIEVED_DOCS", default="False"))
        
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
        self._l2_dist_distance = nn.PairwiseDistance(p=2)
        self._l1_dist_distance = nn.PairwiseDistance(p=1)

        self._vader_analyzer = SentimentIntensityAnalyzer()

    def predict(self, sentences: List[str] = None,  text: str = None, passage: str = None) -> JsonDict:

        return self.predict_json({"sentence": sentences, "text": text, "passage": passage})

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        results = {}

        #results["inputs"] = inputs
        results["passages"] = []

        passages = input_to_passages(inputs, sentence_batch_size=self._sentence_batch_size,
                                     sentence_label_size=self._sentence_label_size,
                                     sentence_step_size=self._sentence_step_size, max_passages=self._max_passages)

        # Set all the metrics to 0.
        for p in passages:
            p["metrics"] = {}
            p["metrics"] ["perplexity"] = 0.0
            p["metrics"] ["sentiment"] = 0.0
            p["metrics"] ["avg_log_likelihood"] = 0.0
            p["metrics"] ["avg_log_likelihood_salience"] = 0.0

            for field in embeddings_fields:
                p["metrics"] [f"{field}_l1_dist"] = 0.0
                p["metrics"][f"{field}_l2_dist"] = 0.0
                p["metrics"][f"{field}_cosine_sim"] = 0.0
                p["metrics"][f"{field}_dot_product"] = 0.0
                p["metrics"][f"{field}_wasserstein_dist"] = 0.0


        # Calculate the offset passages which wkip

        results["passages"] = []

        passages_offset = []
        for example_one, example_two in more_itertools.windowed(passages, n=2):

            if example_two is None:
                continue

            if "label" not in example_one or "label" not in example_two:
                continue

            copied_example_two = copy.deepcopy(example_two)

            copied_example_two["text"] = example_one["text"]

            passages_offset.append(copied_example_two)

        passages_output = []
        passages_offset_output = []
        for p, p_off in zip(passages, passages_offset):

            p_results = self._model_output(p)
            if p_results is not None:
                passages_output.append(p_results)

            p_off_results = self._model_output(p_off)
            if p_off_results is not None:
                passages_offset_output.append(p_off_results)

            if self._add_to_memory:
                self._model.add_to_memory(p["text"], add_to_memory=self._add_to_memory)

        # Map passage output into metrics.
        for p, o in zip(passages, passages_output):
            self._map_output(o, p)
            results["passages"].append(p)

        for p, o in zip(passages_offset, passages_offset_output):
            self._map_output(o, p)

        # Map the vector distance metrics use offsets.
        self.calc_vector_metrics(passages, passages_output)
        results["passages"] = passages
        self.calc_vector_metrics(passages_offset, passages_offset_output)

        # Add the cardinal indexing.

        salience_field_dict =  {"avg_log_likelihood": False}
        for field in embeddings_fields:
            f_dict =  {f"{field}_l1_dist": True,
                        f"{field}_l2_dist": True,
                        f"{field}_cosine_sim": False,
                        f"{field}_dot_product": False,
                        f"{field}_wasserstein_dist": True}

            salience_field_dict = {**salience_field_dict, **f_dict}

        for metric_name, flip in salience_field_dict.items():

            passages = results["passages"]

            passages[0][f"{metric_name}_salience"] = 0.0 # First one is always 0.

            passages_rest = passages[1:]

            for p, p_off in zip(passages_rest, passages_offset):

                if f"{metric_name}" in p["metrics"] and f"{metric_name}" in p_off["metrics"]:
                    one = p["metrics"][f"{metric_name}"]
                    two = p_off["metrics"][f"{metric_name}"]

                    if flip:
                        one = -one
                        two = -two

                    p["metrics"][f"{metric_name}_salience"] = one - two
                else:
                    p["metrics"][f"{metric_name}_salience"] = 0.0

        passages = results["passages"]

        for p in passages:
            p["peaks"] = {}


        peak_field_dict = {"avg_log_likelihood_salience": False,
                     "perplexity": False,
                     "sentiment": False}
        for field in embeddings_fields:
            f_dict = {f"{field}_l1_dist": True,
                      f"{field}_l2_dist": True,
                      f"{field}_cosine_sim": False,
                      f"{field}_dot_product": False,
                      f"{field}_wasserstein_dist": True,
                      f"{field}_l1_dist_salience": False,
                      f"{field}_l2_dist_salience": False,
                      f"{field}_cosine_sim_salience": False,
                      f"{field}_dot_product_salience": False,
                      f"{field}_wasserstein_dist_salience": False
                      }
            peak_field_dict = {**peak_field_dict, **f_dict}

        for k, flip in peak_field_dict.items():

            metric = [m["metrics"][k] for m in passages if k in m["metrics"]]

            if len(metric) == 0:
                continue

            if flip:
                metric = [-m for m in metric]

            # Define ranking for the metric. 0 is top according to the metric.
            sorted_metric_idx = numpy.argsort(metric)
            for i, sor in enumerate(reversed(sorted_metric_idx)):
                passages[sor]["peaks"][f"{k}_rank"] = i

            scaler = MinMaxScaler()
            metric_scaled = numpy.squeeze(scaler.fit_transform(numpy.expand_dims(metric, axis=1)),axis=1)
            ##print(f"Metric scales: {metric_scaled}, {metric}")
            peaks, properties = find_peaks(metric_scaled, prominence=self._peak_prominence, distance=self._peak_distance,
                               threshold=self._peak_threshold, height=self._peak_height)

            ##print(f"Peaks {peaks}, {properties}")

            # Set peak to false
            for p in passages:
                p["peaks"][f"{k}_peak"] = False
                p["peaks"][f"{k}_peak_properties"] = {}

            for i, p in enumerate(peaks):
                passages[p]["peaks"][f"{k}_peak"] = True
                passages[p]["peaks"][f"{k}_peak_properties"]["height"] = float(properties["peak_heights"][i])
                passages[p]["peaks"][f"{k}_peak_properties"]["prominence"] = float(properties["prominences"][i])
                passages[p]["peaks"][f"{k}_peak_properties"]["left_base"] = int(properties["left_bases"][i])
                passages[p]["peaks"][f"{k}_peak_properties"]["right_base"] = int(properties["right_bases"][i])

        self._model.clear_memory()

        # Cleanup the repeating offset and label text.
        for p in results["passages"]:
            del p["label"]

            if "text_offset" in p:
                p["text"] = p["text_offset"]
                del p["text_offset"]

        return results

    def calc_vector_metrics(self, passages, passages_output):
        for i, (first, second) in enumerate(more_itertools.windowed(passages_output, n=2, step=1)):

            if first == None or second == None:
                continue

            first_doc_emb = torch.tensor(first["retrieved_doc_embeddings"])
            second_doc_emb = torch.tensor(second["retrieved_doc_embeddings"])
            metrics = self._vector_distance_metrics("retrieved_doc_embedding", first_doc_emb, second_doc_emb)

            passages[i]["metrics"] = {**passages[i]["metrics"], **metrics}

            first_doc_emb = torch.tensor(first["generator_enc_embeddings"])
            second_doc_emb = torch.tensor(second["generator_enc_embeddings"])
            metrics = self._vector_distance_metrics("generator_enc_embedding", first_doc_emb, second_doc_emb)

            passages[i]["metrics"] = {**passages[i]["metrics"], **metrics, }

            if "generator_dec_embeddings" in first and "generator_dec_embeddings" in second:
                first_doc_emb = torch.tensor(first["generator_dec_embeddings"])
                second_doc_emb = torch.tensor(second["generator_dec_embeddings"])

                metrics = self._vector_distance_metrics("generator_dec_embedding", first_doc_emb, second_doc_emb)
                passages[i]["metrics"] = {**passages[i]["metrics"], **metrics, }

    def _vector_distance_metrics(self, name, x, y):

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

        # #print(f"Normalized: {x}, {y}")

        l1_dist_dist = self._l1_dist_distance(x, y)
        l2_dist_dist = self._l2_dist_distance(x, y)
        cosine_sim = self._cosine_similarity(x, y)
        dot_product = (x * y).sum(-1)

        res_dict[f"{name}_l1_dist"] = l1_dist_dist.item()
        res_dict[f"{name}_l2_dist"] = l2_dist_dist.item()
        res_dict[f"{name}_cosine_sim"] = cosine_sim.item()
        res_dict[f"{name}_dot_product"] = dot_product.item()
        res_dict[f"{name}_wasserstein_dist"] = wass_dist

        return res_dict

    def _map_output(self, output, example):

        print(f"Passages Output Keys, {output.keys()}")

        if "metrics" not in example:
            example["metrics"] = {}
        example["metrics"]["perplexity"] = output["perplexity"].item()
        example["metrics"]["avg_log_likelihood"] = output["avg_log_likelihood"].item()
        example["metrics"]["sentiment"] = output["sentiment"]

        if self._keep_embeddings:
            example["retrieved_doc_embedding"] = output["retrieved_doc_embeddings"].tolist()
            example["generator_enc_embedding"] = output["generator_enc_embeddings"].tolist()
            example["generator_dec_embedding"] = output["generator_dec_embeddings"].tolist()

        if self._add_retrieved_docs and "retrieved_docs" in output:
            example["retreived_docs"] = output["retrieved_docs"]

    def _model_output(self, example):

        if "label" not in example:
            return None

        instance = self._json_to_instance(example)

        outputs = self._model.forward_on_instance(instance)

        sentiment = self._vader_analyzer.polarity_scores(example["text"])
        outputs["sentiment"] = sentiment["compound"]



        return outputs


    def _json_to_instance(self, example: JsonDict) -> Instance:

        fields = {}

        fields["metadata"] = MetadataField(example)
        #logger.info(f"Example: {example}")

        tokens = self.encoder_tokenizer.tokenize(example['text'])

        text_field = TextField(tokens, self.encoder_indexers)
        fields['text'] = text_field

        if "label" in example:
            target_tokens = self.generator_tokenizer.tokenize(example['label'])

            fields['labels'] = TextField(target_tokens, self.generator_indexers)

        return Instance(fields)
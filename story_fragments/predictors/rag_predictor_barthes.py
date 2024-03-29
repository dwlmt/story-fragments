import copy
import os
from itertools import zip_longest
from typing import List

import more_itertools
import nltk
import numpy
import torch
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from story_fragments.predictors.utils import input_to_passages

nltk.download('vader_lexicon')


def parse_bool(b):
    return b == "True" or b == "TRUE" or b == "true" or b == "1"


embeddings_fields = ["retrieved_doc_embedding", "generator_enc_embedding", "generator_dec_embedding"]


@Predictor.register("rag-fragments-barthes")
class RagFragmentsBarthesPredictor(Predictor):
    """ Runs predictions used the barthes cardinal function method.
    """

    def __init__(
            self, model: Model,
            dataset_reader: DatasetReader
    ) -> None:
        super().__init__(model, dataset_reader)

        self._abridge = parse_bool(os.getenv("ABRIDGE", default="False"))
        self._abridge_cycles = int(os.getenv("ABRIDGE_CYCLES", default=4))
        self._abridge_percent = int(os.getenv("ABRIDGE_PERCENT", default=57))
        self._abridge_metric = str(os.getenv("ABRIDGE_METRIC", default="avg_log_likelihood_salience"))
        self._abridge_flip = parse_bool(os.getenv("ABRIDGE_FLIP", default="False"))

        self._sentence_batch_size = int(os.getenv("SENTENCE_BATCH_SIZE", default=12))
        self._sentence_label_size = int(os.getenv("SENTENCE_LABEL_SIZE", default=12))
        self._sentence_step_size = int(os.getenv("SENTENCE_STEP_SIZE", default=4))
        self._max_passages = int(os.getenv("MAX_PASSAGES", default=1000000))

        self._add_to_memory = parse_bool(os.getenv("ADD_TO_MEMORY", default="True"))
        self._keep_embeddings = parse_bool(os.getenv("KEEP_EMBEDDINGS", default="False"))

        generator_model_name = str(os.getenv("GENERATOR_MODEL_NAME", default="facebook/bart-base"))
        generator_max_length = int(os.getenv("GENERATOR_MAX_LENGTH", default=128))
        encoder_model_name = str(os.getenv("ENCODER_MODEL_NAME", default="facebook/dpr-question_encoder-multiset-base"))
        encoder_max_length = int(os.getenv("ENCODER_MAX_LENGTH", default=512))
        add_special_tokens = parse_bool(os.getenv("ADD_SPECIAL_TOKENS", default="True"))

        self._retrieval_metrics = parse_bool(os.getenv("RETRIEVAL_METRICS", default="True"))
        self._random_retrieval = parse_bool(os.getenv("RANDOM_RETRIEVAL", default="False"))
        self._swap_metrics = parse_bool(os.getenv("SWAP_METRICS", default="True"))

        self._calc_vector_metrics = parse_bool(os.getenv("CALC_VECTOR_METRICS", default="True"))

        self._clear_memory_between_instances = parse_bool(os.getenv("CLEAR_MEMORY_BETWEEN_INSTANCES", default="False"))

        self._peak_distance = int(os.getenv("PEAK_DISTANCE", default=5))
        self._peak_prominence = float(os.getenv("PEAK_PROMINENCE", default=0.10))
        self._peak_threshold = float(os.getenv("PEAK_THRESHOLD", default=0.01))
        self._peak_height = float(os.getenv("PEAK_HEIGHT", default=0.01))

        self._cluster_metrics = parse_bool(os.getenv("CLUSTER_METRICS", default="True"))
        self._cluster_ratio = float(os.getenv("CLUSTER_RATIO", default=0.1))
        self._min_clusters = int(os.getenv("MIN_POINTS_PER_CLUSTER", default=5))
        self._cluster_cosine = parse_bool(os.getenv("CLUSTER_COSINE", default="True"))
        self._vector_batch_size = int(os.getenv("VECTOR_BATCH_SIZE", default=20))

        self.sentiment_negative_mixture_weighting = float(
            os.getenv("SENTIMENT_NEGATIVE_MIXTURE_WEIGHTING", default=2.0))
        self.salience_mixture_weighting = float(os.getenv("SALIENCE_MIXTURE_WEIGHTING", default=2.0))

        self._add_retrieved_docs = parse_bool(os.getenv("ADD_RETRIEVED_DOCS", default="False"))

        sentence_transformer_model = str(os.getenv("SENTENCE_TRANSFORMER_MODEL", default='stsb-roberta-large'))
        self.sentence_transformer = SentenceTransformer(sentence_transformer_model).cuda()

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

        # Hacky flag to stop adding duplicates to memory.
        os.environ['DONT_ADD_TO_MEMORY'] = 'True'

        self._previous_text = None

    def predict(self, sentences: List[str] = None, text: str = None, passage: str = None) -> JsonDict:

        return self.predict_json({"sentence": sentences, "text": text, "passage": passage})

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        results = {}

        if "title" in inputs:
            results["title"] = inputs["title"]

        if "id" in inputs:
            results["id"] = inputs["id"]

        if "sentences" in inputs:
            results["input_sentences"] = inputs["sentences"]

        results["passages"] = []

        cycles = self._abridge_cycles if self._abridge else 1
        for i in range(cycles):

            prefill = True if self._sentence_step_size < self._sentence_batch_size else False

            passages = input_to_passages(inputs, sentence_batch_size=self._sentence_batch_size,
                                         sentence_label_size=self._sentence_label_size,
                                         sentence_step_size=self._sentence_step_size, max_passages=self._max_passages,
                                         prefill=prefill)

            if self._abridge and i == 0:
                pass  # results["orig_passages"] = copy.deepcopy(passages)

            # Set all the metrics to 0.
            for p in passages:
                p["metrics"] = {}
                p["metrics"]["perplexity"] = 0.0
                p["metrics"]["sentiment"] = 0.0
                p["metrics"]["sentiment_abs"] = 0.0
                p["metrics"]["avg_log_likelihood"] = 0.0
                p["metrics"]["avg_log_likelihood_salience_impact_adj"] = 0.0

                field_list = ["perplexity", "avg_log_likelihood", "sentiment_abs"]

                if self._retrieval_metrics:
                    for k in field_list:
                        p["metrics"][f"{k}_no_ret"] = 0.0
                        p["metrics"][f"{k}_no_ret_diff"] = 0.0

                if self._swap_metrics:
                    for k in field_list:
                        p["metrics"][f"{k}_swapped"] = 0.0
                        p["metrics"][f"{k}_swapped_diff"] = 0.0

                salience_dict = {}
                for k, v in p["metrics"].items():
                    salience_dict[f"{k}_salience"] = 0.0

                p["metrics"] = {**p["metrics"], **salience_dict}

                if self._calc_vector_metrics:
                    for field in embeddings_fields:
                        p["metrics"][f"{field}_l1_dist"] = 0.0
                        p["metrics"][f"{field}_l2_dist"] = 0.0
                        p["metrics"][f"{field}_cosine_sim"] = 0.0
                        p["metrics"][f"{field}_dot_product"] = 0.0
                        p["metrics"][f"{field}_wasserstein_dist"] = 0.0

            # Calculate the offset passages which wkip

            results["passages"] = []

            passages_offset = self.passage_offsets(passages)
            passages_swapped = self.passage_swapped(passages)

            self._calc_metrics(passages, passages_offset, passages_swapped, results)

            self._calc_salience(results, passages_offset)

            if self._cluster_metrics:
                self.cluster_metrics(results)

            for p in passages:

                sentiment = p["metrics"]["sentiment"]
                if sentiment < 0:
                    sentiment *= self.sentiment_negative_mixture_weighting

                sentiment_adj = (abs(sentiment) + 1.0)

                p["metrics"]["avg_log_likelihood_salience_impact_adj"] = p["metrics"][
                                                                             f"avg_log_likelihood_salience"] * sentiment_adj

            self._calc_peaks(results)

            if self._clear_memory_between_instances:
                self._model.clear_memory()

            self.cleanup_output(results)

            if not self._clear_memory_between_instances:
                self._previous_text = [s["text"] for s in results["passages"]]

            inputs = self.abridge_if_required(inputs, results)

        return results

    def abridge_if_required(self, inputs, results):
        if self._abridge:

            passages_len = len(results["passages"])

            number_to_keep = int((float(self._abridge_percent) / 100.0) * passages_len)
            if not self._abridge_flip:
                retained_passages = [p for p in results["passages"] if
                                     p["peaks"][f"{self._abridge_metric}_rank"] < (passages_len - number_to_keep)]

            else:
                retained_passages = [p for p in results["passages"] if
                                     p["peaks"][f"{self._abridge_metric}_rank"] >= (passages_len - number_to_keep)]

            inputs = {"text": " ".join([p["text"] for p in retained_passages]).replace("<PLACEHOLDER>", "")}
            print(f"ABRIDGED TEXT: {inputs}")
        return inputs

    def _calc_metrics(self, passages, passages_offset, passages_swapped, results):
        passages_output = []
        passages_offset_output = []
        passages_swapped_output = []
        no_kb_output = []
        no_kb_offset_output = []
        # Assume step size is less than the batch size otherwise would skip text.
        add_every = int(self._sentence_batch_size / self._sentence_step_size)
        for i, (p, p_off) in enumerate(zip_longest(passages, passages_offset), start=1):

            if p is not None:
                p_results = self._model_output(p)
                if p_results is not None:
                    passages_output.append(p_results)

                if self._retrieval_metrics:
                    p_copy = copy.deepcopy(p)

                    if not self._random_retrieval:
                        p_copy["ndocs"] = 0

                    no_kb_results = self._model_output(p_copy)
                    if no_kb_results is not None:
                        no_kb_output.append(no_kb_results)

                if self._swap_metrics:
                    p_swapped_results = self._model_output(passages_swapped[i - 1])
                    passages_swapped_output.append(p_swapped_results)

            if p_off is not None:
                p_off_results = self._model_output(p_off)
                if p_off_results is not None:
                    passages_offset_output.append(p_off_results)

                if self._retrieval_metrics:
                    p_copy = copy.deepcopy(p_off)

                    if not self._random_retrieval:
                        p_copy["ndocs"] = 0
                    else:
                        p_copy["ndocs"] = self._model.config.combined_n_docs

                    no_kb_results = self._model_output(p_copy)
                    if no_kb_results is not None:
                        no_kb_offset_output.append(no_kb_results)

            # If a sliding window is used then only add the complete passage to memory.
            if (i % add_every == 0 or add_every == 1) and self._add_to_memory:
                ids, context_embeddings = self._model.add_to_memory(p["text"], add_to_memory=self._add_to_memory)
                # print(ids, context_embeddings)
                memory_id = numpy.asscalar(ids)

                p["memory_id"] = memory_id

                if self._keep_embeddings:
                    p["answer_embedding"] = context_embeddings[0].tolist()

                print(f"ADD TO MEMORY: {ids}, {context_embeddings.shape}")

        # Map passage output into metrics.
        for p, o in zip(passages, passages_output):
            self._map_output(o, p)
            results["passages"].append(p)

        for p, o in zip(passages_offset, passages_offset_output):
            self._map_output(o, p)

        for p, o in zip(passages_offset, passages_offset_output):
            self._map_output(o, p)

        if self._calc_vector_metrics:
            self.calc_vector_metrics(passages, passages_output)
            results["passages"] = passages
            self.calc_vector_metrics(passages_offset, passages_offset_output)

        if self._retrieval_metrics:
            for p, o in zip(passages, no_kb_output):
                self.kb_perplexity_metrics(p, o)

            for p, o in zip(passages_offset, no_kb_offset_output):
                self.kb_perplexity_metrics(p, o)

        if self._swap_metrics:
            for p, o in zip(passages, passages_swapped_output):
                self.swapped_perplexity_metrics(p, o)

        if self._calc_vector_metrics:
            self.calc_vector_metrics(passages, no_kb_output, extension="_no_ret")
            self.calc_vector_metrics(passages_offset, no_kb_offset_output, extension="_no_ret")
        for p in passages:
            self.calc_diff_metrics(p)
        for p in passages_offset:
            self.calc_diff_metrics(p)

    def cluster_metrics(self, results):
        passages = results["passages"]
        embeddings_list = []
        for passage_batch in more_itertools.chunked(passages, n=self._vector_batch_size):
            embeddings_text = []
            for p in passage_batch:
                if "text_offset" in p:
                    text = p["text_offset"]
                else:
                    text = p["text"]

                embeddings_text.append(text)

            embeddings = self.sentence_transformer.encode(embeddings_text)
            embeddings_list.append(embeddings)
        all_embeddings = numpy.concatenate(embeddings_list, axis=0)
        # print(f"All Embeddings: {all_embeddings.shape}")
        num_clusters = max(self._min_clusters, int(len(passages) * self._cluster_ratio))
        if len(passages) > num_clusters:
            kmeans_clusterer = KMeans(n_clusters=num_clusters)

            def distance_from_centroid(assigned_clusters, embeddings, centroids, dist_metric=euclidean):
                distances = []
                for a, e in zip(assigned_clusters, embeddings):
                    a_centroid = centroids[a]
                    distances.append(dist_metric(e, a_centroid))
                return distances

            if self._cluster_cosine:
                from sklearn import preprocessing  # to normalise existing X
                X = preprocessing.normalize(all_embeddings)
            else:
                X = all_embeddings
            assigned_clusters = kmeans_clusterer.fit_predict(X)
            centroids = kmeans_clusterer.cluster_centers_
            # print(f"Assigned Clusters: {assigned_clusters}")
            # print(f"Centroids: {centroids}")
            cluster_distances = distance_from_centroid(assigned_clusters, X, centroids)
            cluster_distances_flipped = [-d for d in cluster_distances]

            for p, d, c in zip(passages, cluster_distances_flipped, assigned_clusters):
                p["metrics"]["cluster"] = c.item()
                p["metrics"]["cluster_score"] = d

                # "avg_log_likelihood""sentiment_abs"

                sentiment = p["metrics"]["sentiment"]
                if sentiment < 0:
                    sentiment *= self.sentiment_negative_mixture_weighting

                sentiment_adj = (abs(sentiment + 1.0))

                p["metrics"]["cluster_score_imp_adj"] = d * sentiment_adj

                # print(p["metrics"])
                p["metrics"]["avg_log_likelihood_salience_cluster"] = p["metrics"]["cluster_score"] + (
                            self.salience_mixture_weighting * p["metrics"]["avg_log_likelihood_salience"])
                p["metrics"]["avg_log_likelihood_salience_cluster_imp_adj"] = ((p["metrics"]["cluster_score"] + (
                            self.salience_mixture_weighting * p["metrics"][
                        "avg_log_likelihood_salience"]))) * sentiment_adj

    def passage_offsets(self, passages):
        passages_offset = []
        for example_one, example_two in more_itertools.pairwise(passages):

            if example_two is None:
                continue

            if "label" not in example_one or "label" not in example_two:
                continue

            copied_example_two = copy.deepcopy(example_two)

            copied_example_two["text"] = example_one["text"]

            passages_offset.append(copied_example_two)
        return passages_offset

    def passage_swapped(self, passages):
        passages_swapped = []
        for example_one in passages:

            if "label" not in example_one:
                continue

            copied_example_one = copy.deepcopy(example_one)

            copied_example_one["text"] = example_one["label"]
            copied_example_one["label"] = example_one["text"]

            passages_swapped.append(copied_example_one)
        return passages_swapped

    def _calc_salience(self, results, passages_offset):
        salience_field_dict = {"avg_log_likelihood": False, "perplexity": False, }
        # Flip (negate) all the distances as with avg_log_likelihood higher should be better.
        for field in embeddings_fields:
            f_dict = {f"{field}_l1_dist": True,
                      f"{field}_l2_dist": True,
                      f"{field}_cosine_sim": False,
                      f"{field}_dot_product": False,
                      f"{field}_wasserstein_dist": True}

            salience_field_dict = {**salience_field_dict, **f_dict}

        if self._retrieval_metrics:
            no_kb_dict = {}
            for k, v in salience_field_dict.items():
                no_kb_dict[f"{k}_no_ret"] = v
                no_kb_dict[f"{k}_no_ret_diff"] = v
            salience_field_dict = {**salience_field_dict, **no_kb_dict}

        if self._swap_metrics:
            swap_dict = {}
            for k, v in salience_field_dict.items():
                swap_dict[f"{k}_swapped"] = v
                swap_dict[f"{k}_swapped_diff"] = v
            salience_field_dict = {**salience_field_dict, **swap_dict}

        for metric_name, flip in salience_field_dict.items():

            passages = results["passages"]

            passages[0][f"{metric_name}_salience"] = 0.0  # First one is always 0.

            for p, p_off in more_itertools.zip_offset(passages, passages_offset, offsets=(1, 0)):

                if f"{metric_name}" in p["metrics"] and f"{metric_name}" in p_off["metrics"]:
                    one = p["metrics"][f"{metric_name}"]
                    two = p_off["metrics"][f"{metric_name}"]

                    if flip:
                        one = -one
                        two = -two

                    p["metrics"][f"{metric_name}_salience"] = one - two

    def _calc_peaks(self, results):
        passages = results["passages"]
        for p in passages:
            p["peaks"] = {}
        peak_field_dict = {"cluster_score": False,
                           "avg_log_likelihood": False,
                           "perplexity": False,
                           "sentiment": False,
                           "sentiment_abs": False,
                           "avg_log_likelihood_salience_impact_adj": False,
                           "avg_log_likelihood_salience_cluster": False,
                           "avg_log_likelihood_salience_cluster_imp_adj": False}
        for field in embeddings_fields:
            f_dict = {f"{field}_l1_dist": False,  # True,
                      f"{field}_l2_dist": False,  # True,
                      f"{field}_cosine_sim": False,
                      f"{field}_dot_product": False,
                      f"{field}_wasserstein_dist": False,  # True,
                      f"{field}_l1_dist_salience": False,
                      f"{field}_l2_dist_salience": False,
                      f"{field}_cosine_sim_salience": False,
                      f"{field}_dot_product_salience": False,
                      f"{field}_wasserstein_dist_salience": False
                      }
            peak_field_dict = {**peak_field_dict, **f_dict}

        if self._retrieval_metrics:
            no_kb_dict = {}
            for k, v in peak_field_dict.items():
                no_kb_dict[f"{k}_no_ret"] = v
                no_kb_dict[f"{k}_no_ret_diff"] = v
            peak_field_dict = {**peak_field_dict, **no_kb_dict}

        if self._swap_metrics:
            swap_dict = {}
            for k, v in peak_field_dict.items():
                swap_dict[f"{k}_swapped"] = v
                swap_dict[f"{k}_swapped_diff"] = v
            peak_field_dict = {**peak_field_dict, **swap_dict}

        salience_dict = {}
        for k, v in peak_field_dict.items():
            if "salience" not in k:
                salience_dict[f"{k}_salience"] = False
        peak_field_dict = {**peak_field_dict, **salience_dict}

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
            metric_scaled = numpy.squeeze(scaler.fit_transform(numpy.expand_dims(metric, axis=1)), axis=1)
            ##print(f"Metric scales: {metric_scaled}, {metric}")
            peaks, properties = find_peaks(metric_scaled, prominence=self._peak_prominence,
                                           distance=self._peak_distance,
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

    def cleanup_output(self, results):
        for p in results["passages"]:
            del p["label"]

            if "text_offset" in p:
                p["text"] = p["text_offset"]
                del p["text_offset"]

        prefill = True if self._sentence_step_size < self._sentence_batch_size else False

        # If a placeholder was used then remove the placeholder and reset the sequence so the first sentence
        # is true first sentence.
        if prefill:
            results["passages"] = results["passages"][1:]
            for i, p in enumerate(results["passages"]):
                p["seq_num"] = i

    def calc_diff_metrics(self, p):
        kb_metrics_dict = {}
        for k, v in p["metrics"].items():
            if "_no_ret" in k:
                continue

            if f"{k}_no_ret" in p["metrics"]:
                kb_metrics_dict[f"{k}_no_ret_diff"] = p["metrics"][f"{k}"] - p["metrics"][f"{k}_no_ret"]
        p["metrics"] = {**kb_metrics_dict, **p["metrics"]}

    def kb_perplexity_metrics(self, passage, output):
        passage["metrics"]["perplexity_no_ret"] = output["perplexity"].item()
        passage["metrics"]["avg_log_likelihood_no_ret"] = output[f"avg_log_likelihood"].item()
        passage["metrics"]["perplexity_no_ret_diff"] = passage["metrics"]["perplexity"] - passage["metrics"][
            "perplexity_no_ret"]
        passage["metrics"]["avg_log_likelihood_no_ret_diff"] = passage["metrics"]["avg_log_likelihood"] - \
                                                               passage["metrics"][
                                                                   "avg_log_likelihood_no_ret"]

    def swapped_perplexity_metrics(self, passage, output):
        passage["metrics"]["perplexity_swapped"] = output["perplexity"].item()
        passage["metrics"]["avg_log_likelihood_swapped"] = output[f"avg_log_likelihood"].item()
        passage["metrics"]["perplexity_swapped_diff"] = passage["metrics"]["perplexity"] - passage["metrics"][
            "perplexity_swapped"]
        passage["metrics"]["avg_log_likelihood_swapped_diff"] = passage["metrics"]["avg_log_likelihood"] - \
                                                                passage["metrics"][
                                                                    "avg_log_likelihood_swapped"]

    def calc_vector_metrics(self, passages, passages_output, extension=""):
        for i, (first, second) in enumerate(more_itertools.pairwise(passages_output)):

            if first == None or second == None:
                continue

            self._calc_embedding_metrics(extension, first, second, i, passages, "retrieved_doc_embeddings",
                                         "retrieved_doc_embedding")
            self._calc_embedding_metrics(extension, first, second, i, passages, "generator_enc_embeddings",
                                         "generator_enc_embedding")
            self._calc_embedding_metrics(extension, first, second, i, passages, "generator_dec_embeddings",
                                         "generator_dec_embedding")
            self._calc_embedding_metrics(extension, first, second, i, passages, "question_embeddings",
                                         "question_embedding")
            self._calc_embedding_metrics(extension, first, second, i, passages, "answer_embeddings",
                                         "answer_embedding")

    def _calc_embedding_metrics(self, extension, first, second, i, passages, output_name, metric_name):
        if output_name in first and output_name in second:
            first_doc_emb = torch.tensor(first[output_name])
            second_doc_emb = torch.tensor(second[output_name])
            metrics = self._vector_distance_metrics(metric_name, first_doc_emb, second_doc_emb,
                                                    extension)

            passages[i]["metrics"] = {**passages[i]["metrics"], **metrics}

    def _vector_distance_metrics(self, name, x, y, extension=""):

        res_dict = {}

        if len(x.size()) == 2:
            x = torch.squeeze(x, dim=0)
            y = torch.squeeze(y, dim=0)

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

        res_dict[f"{name}_l1_dist{extension}"] = l1_dist_dist.item()
        res_dict[f"{name}_l2_dist{extension}"] = l2_dist_dist.item()
        res_dict[f"{name}_cosine_sim{extension}"] = cosine_sim.item()
        res_dict[f"{name}_dot_product{extension}"] = dot_product.item()
        res_dict[f"{name}_wasserstein_dist{extension}"] = wass_dist

        return res_dict

    def _map_output(self, output, example):

        if "metrics" not in example:
            example["metrics"] = {}

        example["metrics"]["perplexity"] = output["perplexity"].item()
        example["metrics"]["avg_log_likelihood"] = output[f"avg_log_likelihood"].item()
        example["metrics"]["sentiment"] = output["sentiment"]
        example["metrics"]["sentiment_abs"] = abs(output["sentiment"])

        if self._keep_embeddings:
            if "retrieved_doc_embeddings" in output:
                example["retrieved_doc_embedding"] = output["retrieved_doc_embeddings"].tolist()

            if "generator_enc_embeddings" in output:
                example["generator_enc_embedding"] = output["generator_enc_embeddings"].tolist()

            if "generator_dec_embeddings" in output:
                example["generator_dec_embedding"] = output["generator_dec_embeddings"].tolist()

            if "question_embeddings" in output:
                example["question_embedding"] = output["question_embeddings"].tolist()

        if "retrieved_docs" in output:
            retrieved_docs = output["retrieved_docs"]

            if not self._add_retrieved_docs:

                for doc in retrieved_docs:
                    if "title" in doc:
                        del doc["title"]

                    if "text" in doc:
                        del doc["text"]

            example["retrieved_docs"] = retrieved_docs
            # example["predicted_text_greedy"] = output["predicted_text"]

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
        # logger.info(f"Example: {example}")

        tokens = self.encoder_tokenizer.tokenize(example['text'])

        text_field = TextField(tokens, self.encoder_indexers)
        fields['text'] = text_field

        if "label" in example:
            target_tokens = self.generator_tokenizer.tokenize(example['label'])

            fields['labels'] = TextField(target_tokens, self.generator_indexers)

        return Instance(fields)

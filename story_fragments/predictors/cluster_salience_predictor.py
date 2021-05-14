import copy
import os
from typing import List

import more_itertools
import nltk
import numpy
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import MetadataField, TextField
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from story_fragments.predictors.utils import input_to_passages

nltk.download('vader_lexicon')


def parse_bool(b):
    return b == "True" or b == "TRUE" or b == "true" or b == "1"


embeddings_fields = ["retrieved_doc_embedding", "generator_enc_embedding", "generator_dec_embedding"]


@Predictor.register("cluster-salience-predictor")
class ClusterSaliencePredictor(Predictor):
    """ Runs predictions used the barthes cardinal function method.
    """

    def __init__(
            self, model: Model,
            dataset_reader: DatasetReader
    ) -> None:
        super().__init__(model, dataset_reader)

        self._abridge = parse_bool(os.getenv("ABRIDGE", default="False"))
        self._abridge_cycles = int(os.getenv("ABRIDGE_CYCLES", default=5))
        self._abridge_percent = int(os.getenv("ABRIDGE_PERCENT", default=25))
        self._abridge_metric = str(os.getenv("ABRIDGE_METRIC", default="avg_log_likelihood_salience"))
        self._abridge_flip = parse_bool(os.getenv("ABRIDGE_FLIP", default="False"))

        self._sentence_batch_size = int(os.getenv("SENTENCE_BATCH_SIZE", default=1))
        self._sentence_label_size = int(os.getenv("SENTENCE_LABEL_SIZE", default=1))
        self._sentence_step_size = int(os.getenv("SENTENCE_STEP_SIZE", default=4))
        self._max_passages = int(os.getenv("MAX_PASSAGES", default=1000000))

        self._peak_distance = int(os.getenv("PEAK_DISTANCE", default=5))
        self._peak_prominence = float(os.getenv("PEAK_PROMINENCE", default=0.10))
        self._peak_threshold = float(os.getenv("PEAK_THRESHOLD", default=0.01))
        self._peak_height = float(os.getenv("PEAK_HEIGHT", default=0.01))

        self._cluster_ratio = float(os.getenv("CLUSTER_RATIO", default=0.1))
        self._min_clusters = int(os.getenv("MIN_POINTS_PER_CLUSTER", default=5))
        self._cluster_cosine = parse_bool(os.getenv("CLUSTER_COSINE", default="True"))
        self._vector_batch_size = int(os.getenv("VECTOR_BATCH_SIZE", default=20))

        sentence_transformer_model = str(os.getenv("SENTENCE_TRANSFORMER_MODEL", default='stsb-roberta-large'))
        self.sentence_transformer = SentenceTransformer(sentence_transformer_model).cuda()

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
                                         prefill=prefill, previous_text=self._previous_text)

            if self._abridge and i == 0:
                pass  # results["orig_passages"] = copy.deepcopy(passages)

            # Set all the metrics to 0.
            for p in passages:
                p["metrics"] = {}
                p["metrics"]["cluster_score"] = 0.0
                p["metrics"]["cluster"] = -1

            results["passages"] = []

            self.cluster_metrics(passages)

            results["passages"] = passages

            self._calc_peaks(results)

            self.cleanup_output(results)

            inputs = self.abridge_if_required(inputs, results)

        return results

    def cluster_metrics(self, passages):
        embeddings_list = []
        for sent_batch in more_itertools.chunked(passages, n=self._vector_batch_size):
            embeddings_text = []
            for sent in sent_batch:
                embeddings_text.append(sent["text"])

            embeddings = self.sentence_transformer.encode(embeddings_text)
            embeddings_list.append(embeddings)
        all_embeddings = numpy.concatenate(embeddings_list, axis=0)
        # print(f"All Embeddings: {all_embeddings.shape}")
        num_clusters = max(self._min_clusters, int(len(passages) * self._cluster_ratio))
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

    def _calc_peaks(self, results):
        passages = results["passages"]
        for p in passages:
            p["peaks"] = {}
        peak_field_dict = {"cluster_score": False}

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

import os
from random import random, choice
from typing import List

import more_itertools
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
from scipy.special import kl_div
from scipy.stats import wasserstein_distance
from torch import nn
from transformers import AutoTokenizer
from blingfire import text_to_sentences

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

        self._input_size = int(os.getenv("SENTENCE_INPUT_SIZE", default=4))
        self._label_size = int(os.getenv("SENTENCE_LABEL_SIZE", default=4))
        self._step_size = int(os.getenv("SENTENCE_STEP_SIZE", default=4))
        self._add_to_memory = parse_bool(os.getenv("ADD_TO_MEMORY", default="True"))
        self._keep_embeddings = parse_bool(os.getenv("KEEP_EMBEDDINGS", default="True"))

        generator_model_name = str(os.getenv("GENERATOR_MODEL_NAME", default="facebook/bart-base"))
        generator_max_length = int(os.getenv("GENERATOR_MAX_LENGTH", default=256))
        encoder_model_name = str(os.getenv("ENCODER_MODEL_NAME", default="facebook/dpr-question_encoder-multiset-base"))
        encoder_max_length = int(os.getenv("ENCODER_MAX_LENGTH", default=256))
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

    def predict(self, sentences: List[str] = None,  text: str = None, passage: str = None) -> JsonDict:

        return self.predict_json({"sentence": sentences, "text": text, "passage": passage})

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        results = {}

        results["inputs"] = inputs
        results["generated"] = []
        
        if "sentences" in inputs and len(inputs["sentences"]) > 0:
            sentences = inputs["sentences"]
        elif  "text" in inputs and len(inputs["text"]) > 0:
            sentences = text_to_sentences(inputs["text"]).split('\n')
        elif "passage" in inputs and len(inputs["passage"]) > 0:
            sentences = text_to_sentences(inputs["text"]).split('\n')
        elif "passages" in inputs and len(inputs["passages"]) > 0:
            passages = list(inputs["passages"])
            sentences = None
        else:
            raise ValueError("Input text or sentences must be provided.")

        #results["inputs"]["sentences"] = [{"seq_num": i,"text": s} for i, s in enumerate(sentences) ]

        #results["inputs"]["passages"] = []
        results["passages"] = []

        example_batches = []
        if sentences is not None:
            windowed_sentences = list(more_itertools.windowed(sentences, n=self._input_size + self._label_size, step=self._step_size
                                                              , fillvalue=" "))

            for i, window in enumerate(windowed_sentences):
                input_text = window[: self._input_size]
                label_text = window[-self._label_size:]

                example = {}
                example["id"] = f"{i}"
                example["text"] = " ".join(input_text).strip()
                example["labels"] = " ".join(label_text).strip()

                example_batches.append(example)

        else:
            for i, (text, label) in enumerate(list(
                more_itertools.windowed(sentences, n=2, step=1, fillvalue=" "))):

                example = {}
                example["id"] = f"{i}"
                example["text"] = text
                example["labels"] = label

                example_batches.append(example)

        model_outputs_list = []
        for example in example_batches:

            instance = self._json_to_instance(example)

            print(f"Instances: {instance}")

            outputs = self._model.forward_on_instance(instance)

            if self._keep_embeddings:
                example["retrieved_doc_embedding"] = outputs["retrieved_doc_embeddings"].tolist()
                example["generator_enc_embedding"] = outputs["generator_enc_embeddings"].tolist()
                #example["question_enc_embedding"] = outputs["question_enc_embeddings"].tolist()

            model_outputs_list.append(outputs)

            results["passages"].append(example)

            #print(f"question_encoder_last_hidden_state: {outputs['question_encoder_last_hidden_state'].size()}")

            if self._add_to_memory:
                self._model.add_to_memory(example["text"],  add_to_memory=self._add_to_memory)

        for i, (first, second) in enumerate(more_itertools.windowed(model_outputs_list, n=2, step=1)):

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

                l1_dist = self._l1_distance(x, y)
                l2_dist = self._l2_distance(x, y)
                cosine_sim = self._cosine_similarity(x, y)
                cosine_dist = 1.0 - cosine_sim
                dot_product = (x * y).sum(-1)

                res_dict[f"{name}_l1_dist"] = l1_dist.item()
                res_dict[f"{name}_l2_dist"] = l2_dist.item()
                res_dict[f"{name}_cosine_sim"] = cosine_sim.item()
                res_dict[f"{name}_cosine_dist"] = cosine_dist.item()
                res_dict[f"{name}_dot_product"] = dot_product.item()
                res_dict[f"{name}_wasserstein_dist"] = wass_dist

                return  res_dict

            first_doc_emb = torch.tensor(first["retrieved_doc_embeddings"])
            second_doc_emb = torch.tensor(second["retrieved_doc_embeddings"])
            metrics = vector_distance_metrics("retrieved_doc_embedding", first_doc_emb, second_doc_emb)
            results["passages"][i]["prediction_metrics"] = metrics

            first_doc_emb = torch.tensor(first["retrieved_doc_embeddings"])
            second_doc_emb = torch.tensor(second["retrieved_doc_embeddings"])
            metrics = vector_distance_metrics("retrieved_doc_embedding", first_doc_emb, second_doc_emb)
            results["passages"][i]["prediction_metrics"] = metrics

            first_doc_emb = torch.tensor(first["generator_enc_embeddings"])
            second_doc_emb = torch.tensor(second["generator_enc_embeddings"])
            metrics = vector_distance_metrics("generator_enc_embedding", first_doc_emb, second_doc_emb)
            results["passages"][i]["prediction_metrics"] = {**metrics,**results["passages"][i]["prediction_metrics"]}

        return results

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
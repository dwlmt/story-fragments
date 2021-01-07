import os
from random import random, choice
from typing import List

import more_itertools
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides
from transformers import AutoTokenizer
from blingfire import text_to_sentences

def parse_bool(b):
    return b == "True" or b == "TRUE" or b == "true" or b == "1"


@Predictor.register("rag-fragments-generation")
class RagFragmentsGenerationPredictor(Predictor):
    """
    Generate text from the RAG fragments model.
    """
    def __init__(
            self, model: Model,
            dataset_reader: DatasetReader
    ) -> None:
        super().__init__(model, dataset_reader)

        self._sentence_batch_size = int(os.getenv("SENTENCE_BATCH_SIZE", default=4))
        self._sentence_step_size = int(os.getenv("SENTENCE_STEP_SIZE", default=4))

        self._length_to_generate = int(os.getenv("GENERATE_LENGTH", default=50))

        self._add_to_memory = parse_bool(os.getenv("ADD_TO_MEMORY", default="True"))
        self._min_length = int(os.getenv("MIN_LENGTH", default=20))
        self._max_length = int(os.getenv("MIN_LENGTH", default=128))
        self._repetition_penalty = float(os.getenv("REPETITION_PENALTY", default=1.0))
        self._num_return_sequences = int(os.getenv("NUM_RETURN_SEQUENCES", default=1))
        self._no_repeat_ngram_size = int(os.getenv("NO_REPEAT_NGRAM_SIZE", default=4))
        self._num_beams = int(os.getenv("NUM_BEAMS", default=1))
        self._num_beam_groups = int(os.getenv("NUM_BEAM_GROUPS", default=1))

        self._top_k = int(os.getenv("TOP_K", default=50))
        self._top_p = float(os.getenv("TOP_P", default=0.9))
        self._temperature =  float(os.getenv("TEMPERATURE", default=1.0))

        self._length_penalty = float(os.getenv("LENGTH_PENALTY", default=1.0))
        self._diversity_penalty = float(os.getenv("DIVERSITY_PENALTY", default=0.5))
        self._do_sample = parse_bool(os.getenv("DO_SAMPLE", default="True"))

        entmax = parse_bool(os.getenv("ENTMAX", default="False"))
        self._model.model.config.entmax = entmax

        entmax_k = int(os.getenv("ENTMAX_K", default=512))
        self._model.model.config.entmax_k = entmax_k

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

        if sentences is not None:
            passages = list(more_itertools.windowed(sentences, n=self._sentence_batch_size, fillvalue=" ", step=self._sentence_step_size))

        passages = [{"seq_num": i, "text": " ".join(s), "prompt": True} for i, s in enumerate(passages)]

        for i, batch in enumerate(passages):
            sentences_joined = batch['text']

            #results["inputs"]["passages"].append(sentences_joined)
            results["generated"].append(batch)

            print(f"{sentences_joined}")

            if i + 1 == len(passages):

                for j in range(1, self._length_to_generate + 1):
                    next_passage = {}
                    generated = self._model.generate(sentences_joined,
                                                     add_to_memory=self._add_to_memory,
                                                     max_length=self._max_length,
                                                     min_length=self._min_length,
                                                     repetition_penalty=self._repetition_penalty,
                                                     num_return_sequences=self._num_return_sequences,
                                                     no_repeat_ngram_size=self._no_repeat_ngram_size,
                                                     num_beams=self._num_beams,
                                                     num_beam_groups=self._num_beam_groups,
                                                     length_penalty=self._length_penalty,
                                                     diversity_penalty=self._diversity_penalty,
                                                     do_sample=self._do_sample,
                                                     top_k=self._top_k,
                                                     top_p=self._top_p,
                                                     temperature=self._temperature
                                                     )

                    if not isinstance(generated, dict):
                        if len(generated) > 1:
                            next_passage["alternatives"] = generated
                        generated = choice(generated)

                    self._model.add_to_memory(generated["text"], add_to_memory=self._add_to_memory)

                    next_passage["seq_num"] = i + j
                    next_passage["text"] = generated["text"]

                    generated["prompt"] = False
                    results["generated"].append(next_passage)

                    sentences_joined = generated["text"]

            elif self._add_to_memory:
                print(f"Add to memory: {sentences_joined}")
                self._model.add_to_memory(sentences_joined,  add_to_memory=self._add_to_memory)

        self._model.clear_memory()

        return results


import os
import re
from random import choice
from typing import List

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

from story_fragments.predictors.utils import input_to_passages


def parse_bool(b):
    return b == "True" or b == "TRUE" or b == "true" or b == "1"


_RE_COMBINE_WHITESPACE = re.compile(r"\s+")


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

        self._sentence_batch_size = int(os.getenv("SENTENCE_BATCH_SIZE", default=12))
        self._sentence_label_size = int(os.getenv("SENTENCE_LABEL_SIZE", default=12))
        self._sentence_step_size = int(os.getenv("SENTENCE_STEP_SIZE", default=4))
        self._max_passages = int(os.getenv("MAX_PASSAGES", default=1000000))

        self._length_to_generate = int(os.getenv("GENERATE_LENGTH", default=100))

        self._add_to_memory = parse_bool(os.getenv("ADD_TO_MEMORY", default="True"))
        self._min_length = int(os.getenv("MIN_LENGTH", default=64))
        self._max_length = int(os.getenv("MAX_LENGTH", default=128))
        self._repetition_penalty = float(os.getenv("REPETITION_PENALTY", default=1.0))
        self._num_return_sequences = int(os.getenv("NUM_RETURN_SEQUENCES", default=5))
        self._num_keep_sequences = int(os.getenv("NUM_KEEP_SEQUENCES", default=1))
        self._no_repeat_ngram_size = int(os.getenv("NO_REPEAT_NGRAM_SIZE", default=4))
        self._num_beams = int(os.getenv("NUM_BEAMS", default=1))
        self._num_beam_groups = int(os.getenv("NUM_BEAM_GROUPS", default=1))

        self._top_k = int(os.getenv("TOP_K", default=50))
        self._top_p = float(os.getenv("TOP_P", default=0.9))
        self._temperature = float(os.getenv("TEMPERATURE", default=1.0))

        self._length_penalty = float(os.getenv("LENGTH_PENALTY", default=1.0))
        self._diversity_penalty = float(os.getenv("DIVERSITY_PENALTY", default=0.5))
        self._do_sample = parse_bool(os.getenv("DO_SAMPLE", default="True"))

    def predict(self, sentences: List[str] = None, text: str = None, passage: str = None) -> JsonDict:

        return self.predict_json({"sentence": sentences, "text": text, "passage": passage})

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        results = {}

        if "title" in inputs:
            results["title"] = inputs["title"]

        if "id" in inputs:
            results["id"] = inputs["id"]

        results["passages"] = []

        passages = input_to_passages(inputs, sentence_batch_size=self._sentence_batch_size,
                                     sentence_label_size=self._sentence_label_size,
                                     sentence_step_size=self._sentence_step_size, max_passages=self._max_passages)

        for i, batch in enumerate(passages):
            sentences_joined = batch['text']

            # results["inputs"]["passages"].append(sentences_joined)
            results["passages"].append(batch)

            print(f"{sentences_joined}")

            if i + 1 == len(passages):

                for j in range(1, self._length_to_generate + 1):
                    next_passage = {}
                    generated = self._model.generate(sentences_joined,
                                                     max_length=self._max_length,
                                                     min_length=self._min_length,
                                                     repetition_penalty=self._repetition_penalty,
                                                     num_return_sequences=self._num_return_sequences,
                                                     num_keep_sequences=self._num_keep_sequences,
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
                    results["passages"].append(next_passage)

                    sentences_joined = generated["text"]

            elif self._add_to_memory:
                print(f"Add to memory: {sentences_joined}")
                self._model.add_to_memory(sentences_joined, add_to_memory=self._add_to_memory)

        self._model.clear_memory()

        story = " ".join([p["text"] for p in results["passages"]])
        results["story"] = _RE_COMBINE_WHITESPACE.sub(" ", story).strip()

        return results

import os
from typing import List

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

from story_fragments.predictors.utils import input_to_passages


def parse_bool(b):
    return b == "True" or b == "TRUE" or b == "true" or b == "1"


@Predictor.register("rag-fragments-generation-alternatives")
class RagFragmentsGenerationAlternativesPredictor(Predictor):
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

        self._add_to_memory = parse_bool(os.getenv("ADD_TO_MEMORY", default="True"))
        self._min_length = int(os.getenv("MIN_LENGTH", default=92))
        self._max_length = int(os.getenv("MAX_LENGTH", default=128))
        self._repetition_penalty = float(os.getenv("REPETITION_PENALTY", default=1.0))
        self._num_return_sequences = int(os.getenv("NUM_RETURN_SEQUENCES", default=5))
        self._num_filtered_sequences = int(os.getenv("NUM_FILTERED_SEQUENCES", default=5))
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

        # results["inputs"] = inputs
        results["passages"] = []

        passages = input_to_passages(inputs, sentence_batch_size=self._sentence_batch_size,
                                     sentence_label_size=self._sentence_label_size,
                                     sentence_step_size=self._sentence_step_size, max_passages=self._max_passages)

        add_every = int(self._sentence_batch_size / self._sentence_step_size)

        for i, batch in enumerate(passages):
            sentences_joined = batch['text']

            print(f"Generate from: {sentences_joined}")

            generated = self._model.generate(sentences_joined,
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

            print(f"Generated: {generated}")
            if not isinstance(generated, dict):
                alternatives = generated
            else:
                alternatives = [generated]

            batch["alternatives"] = alternatives

            print(f"Generated: {batch}")
            results["passages"].append(batch)

            if (i % add_every == 0 or add_every == 1) and self._add_to_memory:
                print(f"Add to memory: {sentences_joined}")
                self._model.add_to_memory(sentences_joined, add_to_memory=self._add_to_memory)

        self._model.clear_memory()

        return results

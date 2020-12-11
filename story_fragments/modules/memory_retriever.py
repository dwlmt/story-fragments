# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RAG Retriever model implementation."""

import re
import time
from typing import List, Tuple

import numpy as np
from more_itertools import chunked
from transformers import logger, BatchEncoding
from transformers.retrieval_rag import CustomHFIndex, RagRetriever, LegacyIndex, LEGACY_INDEX_PATH, CanonicalHFIndex
from transformers.utils import logging

from story_fragments.modules.memory_cache_index import MemoryIndex

logger = logging.get_logger(__name__)

class CustomMemoryHFIndex(CanonicalHFIndex):
    """
    A wrapper around an instance of :class:`~datasets.Datasets`.
    The dataset and the index are both loaded from the indicated paths on disk.

    Args:
        vector_size (:obj:`int`): the dimension of the passages embeddings used by the index
        dataset_path (:obj:`str`):
            The path to the serialized dataset on disk.
            The dataset should have 3 columns: title (str), text (str) and embeddings (arrays of dimension vector_size)
        index_path (:obj:`str`)
            The path to the serialized faiss index on disk.
    """

    def __init__(self, vector_size: int, dataset, index_path=None):
        super().__init__(vector_size, dataset, index_path=index_path)

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        return [self.dataset[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        distances, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs)
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])
        return np.array(ids), np.array(vectors), distances  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def get_doc_dict(self, doc_id: int):

        return self.dataset[doc_id]

class CustomMemoryHFIndex(CustomHFIndex):
    """
    A wrapper around an instance of :class:`~datasets.Datasets`.
    The dataset and the index are both loaded from the indicated paths on disk.

    Args:
        vector_size (:obj:`int`): the dimension of the passages embeddings used by the index
        dataset_path (:obj:`str`):
            The path to the serialized dataset on disk.
            The dataset should have 3 columns: title (str), text (str) and embeddings (arrays of dimension vector_size)
        index_path (:obj:`str`)
            The path to the serialized faiss index on disk.
    """

    def __init__(self, vector_size: int, dataset, index_path=None):
        super().__init__(vector_size, dataset, index_path=index_path)

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        return [self.dataset[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        distances, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs)
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])
        return np.array(ids), np.array(vectors), distances  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def get_doc_dict(self, doc_id: int):

        return self.dataset[doc_id]


class RagMemoryRetriever(RagRetriever):
    """
    Retriever used to get documents from vector queries.
    It retrieves the documents embeddings as well as the documents contents, and it formats them to be used with a RagModel.

    Args:
        config (:class:`~transformers.RagMemoryConfig`):
            The configuration of the RAG model this Retriever is used with. Contains parameters indicating which ``Index`` to build.
            You can load your own custom dataset with ``config.index_name="custom"`` or use a canonical one (default) from the datasets library
            with ``config.index_name="wiki_dpr"`` for example.
        question_encoder_tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer that was used to tokenize the question.
            It is used to decode the question and then use the generator_tokenizer.
        generator_tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer used for the generator part of the RagModel.
        index (:class:`~transformers.retrieval_rag.Index`, optional, defaults to the one defined by the configuration):
            If specified, use this index instead of the one built using the configuration

    """

    _init_retrieval = True

    def __init__(self, config, question_encoder_tokenizer, generator_tokenizer, index=None):

        super().__init__(config, question_encoder_tokenizer, generator_tokenizer, index=index)

        if config.use_memory_retrieval:
            self.memory_index = MemoryIndex(capacity=config.memory_capacity, buffer=config.memory_buffer,
                                            lru=config.memory_lru)

        else:
            self.memory_index = None

        self.use_dataset_retrieval = config.use_dataset_retrieval
        self.use_memory_retrieval = config.use_memory_retrieval

    @staticmethod
    def _build_index(config):
        if config.index_name == "legacy":
            return LegacyIndex(
                config.retrieval_vector_size,
                config.index_path or LEGACY_INDEX_PATH,
            )
        elif config.index_name == "custom":
            return CustomMemoryHFIndex.load_from_disk(
                vector_size=config.retrieval_vector_size,
                dataset_path=config.passages_path,
                index_path=config.index_path,
            )
        else:
            return CanonicalHFIndex(
                vector_size=config.retrieval_vector_size,
                dataset_name=config.dataset,
                dataset_split=config.dataset_split,
                index_name=config.index_name,
                index_path=config.index_path,
                use_dummy_dataset=config.use_dummy_dataset,
            )

    def _main_retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, np.ndarray]:

        ids_batched = []
        vectors_batched = []
        distances_batched = []
        source_batched = []

        if self.use_dataset_retrieval:
            for question_hidden_states in self._chunk_tensor(question_hidden_states, self.batch_size):
                start_time = time.time()

                ids, vectors, distances = self.index.get_top_docs(question_hidden_states, self.config.n_docs)
                logger.debug(f"Dataset retrieval: {ids}, {vectors.shape}, {distances}")
                logger.debug(
                    "index search time: {} sec, batch size {}".format(
                        time.time() - start_time, question_hidden_states.shape
                    )
                )

                empty = np.where(ids == -1)[0]

                if len(empty) == 0:
                    ids_batched.append(ids)
                    vectors_batched.append(vectors)
                    source_batched.append(np.ones(ids.shape, dtype=np.int))
                    distances_batched.append(distances)

        if self.use_memory_retrieval:
            for question_hidden_states in self._chunk_tensor(question_hidden_states, self.batch_size):
                start_time = time.time()

                memory_ids, memory_vectors, memory_distances = self.memory_index.get_top_docs(question_hidden_states,
                                                                                              self.config.memory_n_docs)
                logger.debug(f"Memory retrieval: {memory_ids}, {memory_vectors.shape}, {memory_distances}")
                logger.debug(
                    "memory index search time: {} sec, batch size {}".format(
                        time.time() - start_time, question_hidden_states.shape
                    )
                )

                empty = np.where(memory_ids == -1)[0]

                if len(empty) == 0:
                    ids_batched.append(memory_ids)
                    vectors_batched.append(memory_vectors)

                    source_batched.append(np.zeros(memory_ids.shape, dtype=np.int))
                    distances_batched.append(memory_distances)

        # print(f"Ids batched: {ids_batched}")

        if self.use_memory_retrieval and self.use_dataset_retrieval:
            if len(ids_batched) == 1:
                ids_arr = np.array(ids_batched[0])
                vectors_arr = np.array(vectors_batched[0])
                distances_arr = np.array(distances_batched[0])
                sources_arr = np.array(source_batched[0])

            else:
                ids_arr = np.concatenate([np.array(a) for a in ids_batched], axis=1)
                vectors_arr = np.concatenate([np.array(a) for a in vectors_batched], axis=1)
                distances_arr = np.concatenate([np.array(a) for a in distances_batched], axis=1)
                sources_arr = np.concatenate([np.array(a) for a in source_batched], axis=1)

            # print(f"Ids concat: {ids_arr}")

            if ids_arr.shape[1] > self.config.combined_n_docs:
                sorted_indices = np.argsort(-(distances_arr), axis=1)

                ids_arr = np.take_along_axis(ids_arr, sorted_indices, axis=1)
                distances_arr = np.take_along_axis(distances_arr, sorted_indices, axis=1)
                vectors_arr = np.take_along_axis(vectors_arr, np.expand_dims(sorted_indices, axis=2), axis=1)
                sources_arr = np.take_along_axis(sources_arr, sorted_indices, axis=1)

                # print(f"Sorted: {ids_arr}, {distances_arr}")

                ids_arr = ids_arr[:, 0: n_docs]
                distances_arr = distances_arr[:, 0: n_docs]
                vectors_arr = vectors_arr[:, 0: n_docs]
                sources_arr = sources_arr[:, 0: n_docs]

        else:

            ids_arr = np.array(ids_batched[0])
            vectors_arr = np.array(vectors_batched[0])
            distances_arr = np.array(distances_batched[0])
            sources_arr = np.array(source_batched[0])

        # print(f"Truncated: {ids_arr}, {distances_arr}")

        return (
            ids_arr,
            vectors_arr,
            distances_arr,
            sources_arr
        )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, List[dict]]:
        """
        Retrieves documents for specified ``question_hidden_states``.

        Args:
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`):
                A batch of query vectors to retrieve with.
            n_docs (:obj:`int`):
                The number of docs retrieved per query.

        Return:
            :obj:`Tuple[np.ndarray, np.ndarray, List[dict]]`:
            A tuple with the following objects:

            - **retrieved_doc_embeds** (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs, dim)`) -- The
              retrieval embeddings of the retrieved docs per query.
            - **doc_ids** (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs)`) -- The ids of the documents in the
              index
            - **doc_dicts** (:obj:`List[dict]`): The :obj:`retrieved_doc_embeds` examples per query.
        """
        doc_ids, retrieved_doc_embeds, distances, sources = self._main_retrieve(question_hidden_states, n_docs)

        doc_dicts = []
        # print(f"Doc ids: {doc_ids}")
        for doc, source in zip(doc_ids, sources):
            for d, s in zip(doc, source):

                if int(s) == int(1):
                    # print(f"Get from dataset: {d}, {s}")
                    doc_dict = self.index.get_doc_dict(int(d))
                    # print(f"Dataset doc dict: {d}, {s}")
                else:
                    # print(f"Get from memory: {d}, {s}")
                    doc_dict = self.memory_index.get_doc_dict(int(d))
                    # print(f"Memory doc dict: {d}, {s}")

                doc_dicts.append(doc_dict)

        # print(f"Retrieved doc_dicts: {doc_dicts}")

        joined_dicts = []
        for doc_chunks in chunked(doc_dicts, self.config.combined_n_docs):
            joined_doc_dict = {}
            joined_doc_dict['id'] = [d['id'] for d in doc_chunks]
            joined_doc_dict['text'] = [d['text'] for d in doc_chunks]
            joined_doc_dict['title'] = [d['title'] for d in doc_chunks]
            joined_doc_dict['embeddings'] = np.array([d['embeddings'] for d in doc_chunks])
            joined_dicts.append(joined_doc_dict)

        # print(f"Joined doc_dicts: {len(joined_dicts)}, {joined_dicts}")

        # doc_dicts = self.index.get_doc_dicts(doc_ids)
        ##print(f"Original doc_dicts: {doc_dicts}")

        return retrieved_doc_embeds, doc_ids, joined_dicts

    def postprocess_docs(self, docs, input_strings, prefix, n_docs, return_tensors=None):
        r"""
        Postprocessing retrieved ``docs`` and combining them with ``input_strings``.

        Args:
            doc_scores (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs)`):
                Retrieval scores of respective docs - passed for logging.
            docs  (:obj:`dict`):
                Retrieved documents.
            input_strings (:obj:`str`):
                Input strings decoded by ``preprocess_query``.
            prefix (:obj:`str`):
                Prefix added at the beginning of each input, typically used with T5-based models.

        Return:
            :obj:`tuple(tensors)`:
                a tuple consisting of two elements: contextualized ``input_ids`` and a compatible ``attention_mask``.
        """

        _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

        def cat_input_and_doc(doc_title, doc_text, input_string, prefix, skip_title=True):

            if doc_title.startswith('"'):
                doc_title = doc_title[1:]
            if doc_title.endswith('"'):
                doc_title = doc_title[:-1]

            # The Wikiplots dataset prefixes the title with a unique id using a colon. Remove if present.
            id_end = doc_title.find(":")
            if id_end != -1:
                doc_title = doc_title[id_end:]

            title_sep = self.config.title_sep
            if skip_title:
                doc_title = ""
                title_sep = ""
                
            if prefix is None:
                prefix = ""
            out = (f"{prefix} {doc_title} {title_sep} {doc_text} {self.config.doc_sep} {input_string}")
            out = _RE_COMBINE_WHITESPACE.sub(" ", out).strip()

            return out

        # print(f"Indices {len(docs)}, {n_docs}, {len(input_strings)}, {len( docs[0]['title'])},  {docs[0]['text']}")
        rag_input_strings = [
            cat_input_and_doc(
                docs[i]["title"][j],
                docs[i]["text"][j],
                input_strings[i],
                prefix,
            )
            for i in range(min(len(docs), len(input_strings)))
            for j in range(n_docs)
        ]

        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
            rag_input_strings,
            max_length=self.config.max_combined_length,
            return_tensors=return_tensors,
            padding="max_length",
            truncation=True,
        )
        # logger.info(f"Contextualised inputs: {torch.sum((contextualized_inputs['input_ids'] != 1), dim=-1)}, {contextualized_inputs['input_ids'].size()} ")

        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]

    def __call__(
            self,
            question_input_ids: List[List[int]],
            question_hidden_states: np.ndarray,
            prefix=None,
            n_docs=None,
            return_tensors=None,
    ) -> BatchEncoding:
        """
        Retrieves documents for specified :obj:`question_hidden_states`.

        Args:
            question_input_ids: (:obj:`List[List[int]]`) batch of input ids
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            prefix: (:obj:`str`, `optional`):
                The prefix used by the generator's tokenizer.
            n_docs (:obj:`int`, `optional`):
                The number of docs retrieved per query.
            return_tensors (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`, defaults to "pt"):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.

        Output:
            :class:`~transformers.BatchEncoding`: A :class:`~transformers.BatchEncoding` with the following fields:

            - **context_input_ids** -- List of token ids to be fed to a model.

              `What are input IDs? <../glossary.html#input-ids>`__
            - **context_attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              :obj:`return_attention_mask=True` or if `"attention_mask"` is in :obj:`self.model_input_names`).

              `What are attention masks? <../glossary.html#attention-mask>`__
            - **retrieved_doc_embeds** -- List of embeddings of the retrieved documents
            - **doc_ids** -- List of ids of the retrieved documents
        """

        n_docs = n_docs if n_docs is not None else self.config.combined_n_docs
        prefix = prefix if prefix is not None else self.config.generator.prefix

        # print(f"Question hidden states:", question_hidden_states.shape)
        retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states, n_docs)

        # print(f"Question input ids: {question_input_ids.size()}")
        input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)
        # print(f"Input strings: {len(input_strings)}")
        context_input_ids, context_attention_mask = self.postprocess_docs(
            docs, input_strings, prefix, n_docs, return_tensors=return_tensors
        )

        return BatchEncoding(
            {
                "context_input_ids": context_input_ids,
                "context_attention_mask": context_attention_mask,
                "retrieved_doc_embeds": retrieved_doc_embeds,
                "doc_ids": doc_ids,
            },
            tensor_type=return_tensors,
        )

    def add(self, context_dicts: List[dict], context_hidden_states: np.ndarray):
        """ Add vectors and dicts to the index.
        Args:
            context_dicts (List[dict]): A list of dictionaries with the representations. Must contain id, title and text fields.
            context_hidden_states (ndarray): The ndarray is batch size * dim.
        """
        if self.memory_index is not None:
            self.memory_index.add(context_dicts=context_dicts, context_hidden_states=context_hidden_states)

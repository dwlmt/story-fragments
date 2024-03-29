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
""" RAG model configuration """

from transformers.models.rag import RagConfig


class RagMemoryConfig(RagConfig):
    model_type = "rag"
    is_composition = True

    def __init__(
            self,
            vocab_size=None,
            is_encoder_decoder=True,
            prefix=None,
            bos_token_id=None,
            pad_token_id=None,
            eos_token_id=None,
            decoder_start_token_id=None,
            title_sep=" / ",
            doc_sep=" // ",
            n_docs=5,
            max_combined_length=300,
            retrieval_vector_size=768,
            retrieval_batch_size=32,
            dataset="wiki_dpr",
            dataset_split="train",
            embeddings_name="multiset",
            index_name="compressed",
            index_path=None,
            passages_path=None,
            use_dummy_dataset=False,
            reduce_loss=False,
            label_smoothing=0.0,
            do_deduplication=True,
            exclude_bos_score=False,
            do_marginalize=False,
            output_retrieved=False,
            use_dataset_retrieval=True,
            use_memory_retrieval=True,
            rag_text_concat_first=False,
            memory_n_docs: int = 5,
            memory_capacity: int = 19000,
            memory_buffer=1000,
            memory_lru: bool = True,
            combined_n_docs: int = 5,
            context_encoder: str = "facebook/dpr-ctx_encoder-multiset-base",
            unlikelihood_ratio: float = 1.0,
            unlikelihood_beta: float = 0.5,
            entmax: bool = False,
            entmax_k: int = 512,
            memory_retrieval_weighting: float = 1.0,
            train_context_encoder: bool = False,
            max_doc_context_length: int = 256,
            **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=is_encoder_decoder,
            prefix=prefix,
            vocab_size=vocab_size,
            title_sep=title_sep,
            doc_sep=doc_sep,
            n_docs=n_docs,
            max_combined_length=max_combined_length,
            retrieval_vector_size=retrieval_vector_size,
            retrieval_batch_size=retrieval_batch_size,
            dataset=dataset,
            dataset_split=dataset_split,
            index_name=index_name,
            index_path=index_path,
            passages_path=passages_path,
            use_dummy_dataset=use_dummy_dataset,
            reduce_loss=reduce_loss,
            label_smoothing=label_smoothing,
            do_deduplication=do_deduplication,
            exclude_bos_score=exclude_bos_score,
            do_marginalize=do_marginalize,
            output_retrieved=output_retrieved,
            **kwargs,
        )

        self.embeddings_name = embeddings_name
        #self.use_dataset_retrieval = use_dataset_retrieval REDUNDANT - Kept so can reuse old models.
        #self.use_memory_retrieval = use_memory_retrieval REDUNDANT - Kept so can reuse old models.
        self.rag_text_concat_first = rag_text_concat_first
        self.memory_n_docs = memory_n_docs
        self.memory_capacity = memory_capacity
        self.memory_buffer = memory_buffer
        self.memory_lru = memory_lru
        self.combined_n_docs = combined_n_docs
        self.context_encoder = context_encoder
        self.unlikelihood_ratio = unlikelihood_ratio
        self.unlikelihood_beta = unlikelihood_beta
        self.entmax = entmax
        self.entmax_k = entmax_k
        self.memory_retrieval_weighting = memory_retrieval_weighting
        self.train_context_encoder = train_context_encoder
        self.max_doc_context_length = max_doc_context_length

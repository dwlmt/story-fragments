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
"""RAG model implementation."""
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple

import torch
from entmax import Entmax15Loss, entmax15, EntmaxBisectLoss, entmax_bisect
from torch.nn import functional as F
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, \
    PretrainedConfig, PreTrainedModel, RagConfig, BeamSearchScorer, LogitsProcessorList, BeamScorer
from transformers.file_utils import ModelOutput
from transformers.models.rag.modeling_rag import RetrievAugLMOutput, RagModel, RagTokenForGeneration, \
    RetrievAugLMMarginOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)

PAD_VALUE = 1


def div(x, y):
    if y == 0:
        return x
    else:
        return x / y

@dataclass
class RetrieveAugMemLMMarginOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    perplexity: Optional[torch.FloatTensor] = None
    avg_log_likelihood: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    doc_scores: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    retrieved_doc_embeds: Optional[torch.FloatTensor] = None
    retrieved_doc_ids: Optional[torch.LongTensor] = None
    context_input_ids: Optional[torch.LongTensor] = None
    context_attention_mask: Optional[torch.LongTensor] = None
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    question_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = None
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor]] = None

class NGramIterator:
    """ N-Gram iterator for a list.
        Based on the one from ParlAI - https://github.com/facebookresearch/ParlAI/blob/fd1b8bb565a1a27bcc8326a36afde03a963819ba/projects/dialogue_unlikelihood/agents.py
    """

    def __init__(self, lst, n):
        self.lst = lst
        self.n = n
        self.max = len(lst) - n

    def __iter__(self):
        self.counter = -1
        return self

    def __next__(self):
        self.counter += 1
        if self.counter > self.max:
            raise StopIteration
        return tuple(self.lst[self.counter: self.counter + self.n])


class RagMemoryModel(RagModel):
    def __init__(
            self,
            config: Optional[PretrainedConfig] = None,
            question_encoder: Optional[PreTrainedModel] = None,
            generator: Optional[PreTrainedModel] = None,
            retriever: Optional = None,  # or maybe just use a `set_retriever(...)` method
            **kwargs,
    ):
        super().__init__(config=config,
                         question_encoder=question_encoder,
                         generator=generator,
                         retriever=retriever,
                         **kwargs)

        self.use_dataset_retrieval = config.use_dataset_retrieval
        self.use_memory_retrieval = config.use_memory_retrieval
        self.combined_n_docs = config.combined_n_docs

        self.context_encoder = DPRContextEncoder.from_pretrained(config.context_encoder)
        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(config.context_encoder)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            past_key_values=None,
            doc_scores=None,
            context_input_ids=None,
            context_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            output_retrieved=None,
            n_docs=None,
            input_text_metadata=None,
    ):
        r"""
        """
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_retrieved = output_retrieved if output_retrieved is not None else self.config.output_retrieved

        # whether retriever has to be used
        has_to_retrieve = (
                self.retriever is not None
                and (context_input_ids is None or context_attention_mask is None or doc_scores is None)
                and encoder_outputs is None
                and n_docs > 0
        )
        # encoder_outputs are pre-computed during RAG-token generation
        if encoder_outputs is None:

            if has_to_retrieve:

                question_enc_outputs = self.question_encoder(
                    input_ids, attention_mask=attention_mask, return_dict=True
                )
                question_encoder_last_hidden_state = question_enc_outputs[0]  # hidden states of question encoder

                retriever_outputs = self.retriever(
                    input_ids,
                    question_encoder_last_hidden_state.cpu().detach().to(torch.float32).numpy(),
                    prefix=self.generator.config.prefix,
                    n_docs=n_docs,
                    return_tensors="pt",
                )

                n_docs = abs(n_docs)

                context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids,\
                    doc_input_ids, doc_attention_mask = (
                    retriever_outputs["context_input_ids"],
                    retriever_outputs["context_attention_mask"],
                    retriever_outputs["retrieved_doc_embeds"],
                    retriever_outputs["doc_ids"],
                    retriever_outputs["doc_input_ids"],
                    retriever_outputs["doc_attention_mask"]
                )

                # set to correct device
                retrieved_doc_embeds = retrieved_doc_embeds.to(question_encoder_last_hidden_state)
                context_input_ids = context_input_ids.to(input_ids)
                context_attention_mask = context_attention_mask.to(input_ids)

                # If training the context encoder then re-encode as needed for a gradient.
                if self.config.train_context_encoder and self.training:
                    doc_input_ids = doc_input_ids.to(input_ids)
                    doc_attention_mask = doc_attention_mask.to(input_ids)
                    self.encode_context_embeddings(doc_input_ids, doc_attention_mask)

                # compute doc_scores
                doc_scores = torch.bmm(
                    question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                ).squeeze(1)

                #if self.use_memory_retrieval:
                if not 'DONT_ADD_TO_MEMORY' in os.environ or os.environ.get("DONT_ADD_TO_MEMORY") != "True":
                    self.add_to_memory(input_ids, attention_mask, input_text_metadata)

            else:
                assert (
                        context_input_ids is not None
                ), "Make sure that `context_input_ids` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."
                assert (
                        context_attention_mask is not None
                ), "Make sure that `context_attention_mask` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."
                assert (
                        doc_scores is not None
                ), "Make sure that `doc_scores` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."

        #n_docs =  min(n_docs, doc_scores.size()[1])
        actual_n_docs = n_docs #doc_scores.shape[1]

        # Decoder input without context documents
        if decoder_input_ids is not None and actual_n_docs > 1:
            decoder_input_ids = decoder_input_ids.repeat_interleave(actual_n_docs, dim=0)

        if decoder_attention_mask is not None and actual_n_docs > 1:
            decoder_attention_mask = decoder_attention_mask.repeat_interleave(actual_n_docs, dim=0)

        gen_outputs = self.generator(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            #output_attentions=output_attentions,
            output_hidden_states=True
        )

        if not has_to_retrieve:
            question_encoder_last_hidden_state = None
            question_enc_hidden_states = None
            question_enc_attentions = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None
        else:
            question_enc_hidden_states = question_enc_outputs.hidden_states
            question_enc_attentions = question_enc_outputs.attentions

        if not has_to_retrieve or not output_retrieved:
            # don't output retrieved docs
            context_input_ids = (None,)
            context_attention_mask = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None

        return RetrievAugLMOutput(
            logits=gen_outputs.logits,
            doc_scores=doc_scores,
            past_key_values=gen_outputs.past_key_values,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            retrieved_doc_embeds=retrieved_doc_embeds,
            retrieved_doc_ids=retrieved_doc_ids,
            question_encoder_last_hidden_state=question_encoder_last_hidden_state,
            question_enc_hidden_states=question_enc_hidden_states,
            question_enc_attentions=question_enc_attentions,
            generator_enc_last_hidden_state=gen_outputs.encoder_last_hidden_state,
            generator_enc_hidden_states=gen_outputs.encoder_hidden_states,
            generator_enc_attentions=gen_outputs.encoder_attentions,
            generator_dec_hidden_states=gen_outputs.decoder_hidden_states,
            generator_dec_attentions=gen_outputs.decoder_attentions,
            #generator_cross_attentions=gen_outputs.cross_attentions
        )

    def add_to_memory(self, input_ids, attention_mask, input_text_metadata):
        with torch.no_grad():

            context_embeddings = self.encode_context_embeddings(input_ids, attention_mask)

            ids = self.retriever.add(context_dicts=input_text_metadata, context_hidden_states=context_embeddings)
            return ids, context_embeddings

    def encode_context_embeddings(self, input_ids, attention_mask):
        ctx_enc_outputs = self.context_encoder(
            input_ids, attention_mask=attention_mask, return_dict=True
        )

        context_embeddings = ctx_enc_outputs.pooler_output.detach().cpu().to(torch.float32).numpy()

        return context_embeddings

    def clear_memory(self):

        self.retriever.clear_memory()

class RagMemoryTokenForGeneration(RagTokenForGeneration):
    def __init__(
            self,
            config: Optional[PretrainedConfig] = None,
            question_encoder: Optional[PreTrainedModel] = None,
            generator: Optional[PreTrainedModel] = None,
            retriever: Optional = None,
            **kwargs,
    ):
        assert config is not None or (
                question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_encoder_generator_configs(question_encoder.config, generator.config, **kwargs)

        super().__init__(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

        # instantiate model
        self.rag = RagMemoryModel(config=config, question_encoder=question_encoder, generator=generator,
                                  retriever=retriever)

        if self.config.entmax:
            pass #self.entmax_alpha = torch.tensor(1.5, requires_grad=True).half()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            past_key_values=None,
            context_input_ids=None,
            context_attention_mask=None,
            doc_scores=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            output_retrieved=None,
            do_marginalize=None,
            reduce_loss=None,
            labels=None,
            n_docs=None,
            input_text_metadata=None,
            **kwargs  # needs kwargs for generation
    ):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_marginalize = do_marginalize if do_marginalize is not None else self.config.do_marginalize
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        outputs = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
            input_text_metadata=input_text_metadata
        )

        n_docs = abs(n_docs)

        loss = None
        perplexity = None
        avg_ll = None
        logits = outputs.logits
        doc_scores = outputs.doc_scores
        context_input_ids = outputs.context_input_ids
        if labels is not None:
            assert decoder_input_ids is not None

            n_docs = abs(n_docs)
            actual_n_docs = n_docs

            rag_logprobs = self.marginalize(logits, doc_scores, actual_n_docs)

            # #print(f"nll input: {outputs.logits.size()}, {outputs.doc_scores.size()}, {labels.size()} ")
            loss, perplexity, avg_ll = self.get_nll(
                rag_logprobs,
                doc_scores,
                labels,
                reduce_loss=reduce_loss,
                epsilon=self.config.label_smoothing,
                n_docs=n_docs,
            )

            if (torch.rand(1).item() >= self.config.unlikelihood_ratio):
                unlikelihood_loss = self.get_unlikelihood_loss(
                    rag_logprobs=rag_logprobs,
                    context_input_ids=context_input_ids,
                    labels=labels,
                    unlikelihood_beta=self.config.unlikelihood_beta
                )
                # print(f"Unlikelihood loss: {unlikelihood_loss}")
                loss += unlikelihood_loss

        if do_marginalize:
            actual_n_docs = n_docs #doc_scores.shape[1]
            logits = self.marginalize(logits, outputs.doc_scores, actual_n_docs)

        return RetrieveAugMemLMMarginOutput(
            loss=loss,
            perplexity=perplexity,
            avg_log_likelihood=avg_ll,
            logits=logits,
            doc_scores=outputs.doc_scores,
            past_key_values=outputs.past_key_values,
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
            #generator_cross_attentions=outputs.generator_cross_attentions,
        )

    def get_nll(self, rag_logprobs, doc_scores, target, reduce_loss=False, epsilon=0.0, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        # shift tokens left
        target = torch.cat(
            [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
        )

        def _mask_pads(ll):#, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                #smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1)#, smooth_obj.squeeze(-1)

        target = target.unsqueeze(-1)
        assert target.dim() == rag_logprobs.dim()

        if not self.config.entmax:
            ll = rag_logprobs.gather(dim=-1, index=target)
            #smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits
            #ll, smooth_obj = _mask_pads(ll, smooth_obj)
            ll = _mask_pads(ll)
            ll = ll.sum(1)  # sum over tokens
            # smooth_obj = smooth_obj.sum(1)

            nll_loss = -ll
            # smooth_loss = -smooth_obj

            #if reduce_loss:
                #nll_loss = nll_loss.sum()
                # smooth_loss = smooth_loss.sum()

            #eps_i = epsilon / rag_logprobs.size(-1)
            loss = nll_loss #(1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        else:
            entmax_loss = Entmax15Loss(k=self.config.entmax_k, ignore_index=self.config.pad_token_id,
                                           reduction="sum")

            target = torch.squeeze(target, dim=2)

            rag_probs = torch.exp(rag_logprobs)

            loss = entmax_loss(rag_probs.view(rag_logprobs.size()[0] * rag_logprobs.size()[1], -1),
                               target.view(target.size()[0] * target.size()[1]))

        with torch.no_grad():
            #target = torch.squeeze(target, dim=0)
            label_mask = (target != self.config.pad_token_id)
            #print(f"Label mask: {label_mask}")
            n_tokens = torch.sum(
                label_mask.view(label_mask.size()[0] * label_mask.size()[1]))

            # Calculate perplexity
            perplexity = torch.exp(loss / n_tokens)# / n_docs)

            avg_ll = (-loss / n_tokens) #/ n_docs)

        return loss, perplexity, avg_ll

    def count_n_grams(self, tokens, n):
        n_grams = defaultdict(int)
        for n_gram in NGramIterator(tokens, n):
            n_grams[n_gram] += 1
        return n_grams

    def get_unlikelihood_loss(self, rag_logprobs, context_input_ids, labels, unlikelihood_ngrams: int = 4,
                              unlikelihood_beta: float = 0.5):

        pred_tokens = torch.max(rag_logprobs, dim=-1)[1]

        ##print(f"Unlikelihood training: {rag_logprobs.size()}, {pred_tokens.size()}, {context_input_ids.size()}")

        crep_mask = torch.zeros_like(pred_tokens).type_as(rag_logprobs)
        lrep_mask = torch.zeros_like(pred_tokens).type_as(rag_logprobs)

        # print(f"Unlikelihood: {pred_tokens.size()}, {rag_logprobs.size()}, {context_input_ids}")
        for i, (tokens, logprob, context, lab) in enumerate(zip(pred_tokens, rag_logprobs, context_input_ids, labels)):

            if context is not None:
                context_ids_list = context.cpu().detach().tolist()
            else:
                context_ids_list = []

            labels_ids_list = lab.cpu().detach().tolist()

            context_n_grams = self.count_n_grams(context_ids_list, n=unlikelihood_ngrams)

            ##print(f"Ngrams: {context_ngrams}")

            seen_n_grams = defaultdict(int)

            # penalize if there is a context repeat
            tokens_id_list = tokens.cpu().detach().tolist()
            for j, n_gram in enumerate(NGramIterator(tokens_id_list, unlikelihood_ngrams)):
                if context_n_grams[n_gram] > 0 and n_gram != tuple(labels_ids_list[j: j + unlikelihood_ngrams]):
                    # print(f"Context seen: {n_gram}")
                    crep_mask[i, j: j + unlikelihood_ngrams] = 1

            for j, n_gram in enumerate(NGramIterator(tokens_id_list, unlikelihood_ngrams)):
                if seen_n_grams[n_gram] > 0 and n_gram != tuple(labels_ids_list[j: j + unlikelihood_ngrams]):
                    # print(f"Label seen: {n_gram}")
                    lrep_mask[i, j: j + unlikelihood_ngrams] = 1
                seen_n_grams[n_gram] += 1

            # print(f"Context Ngrams: {context_n_grams}")
            # print(f"Seen Ngrams: {seen_n_grams}")

        pred_lprobs = rag_logprobs.view(-1, rag_logprobs.size(2)).gather(1, pred_tokens.view(-1, 1).long())
        # print(f"pred_lprobs: {rag_logprobs}, {pred_tokens}")

        one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=1e-6).view(
            pred_tokens.size(0), pred_tokens.size(1)
        )

        # print(f"Masks sum: {torch.sum(lrep_mask, dim=-1)},  {torch.sum(crep_mask, dim=-1)}")
        mask = ((1 - unlikelihood_beta) * lrep_mask) + (
                unlikelihood_beta * crep_mask
        )

        ul_loss = -(torch.log(one_minus_probs)) * mask
        # print(f"ul loss: {ul_loss}")
        total_loss = ul_loss.sum()  # div(ul_loss.sum(), mask.sum())

        return total_loss

    def marginalize(self, seq_logits, doc_scores, n_docs=None):

        n_docs = n_docs if n_docs is not None else self.config.combined_n_docs

        if n_docs == 0:
            n_docs = 1

        print(seq_logits.size(), doc_scores.size(), n_docs)
        seq_logprobs = torch.nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )

        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        return torch.logsumexp(log_prob_sum, dim=1)

    @torch.no_grad()
    def generate(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            context_input_ids=None,
            context_attention_mask=None,
            doc_scores=None,
            max_length=None,
            min_length=None,
            do_sample=None,
            early_stopping=None,
            num_beams=None,
            temperature=None,
            top_k=None,
            top_p=None,
            use_cache=None,
            num_beam_groups=None,
            diversity_penalty=None,
            bos_token_id=None,
            pad_token_id=None,
            eos_token_id=None,
            length_penalty=None,
            no_repeat_ngram_size=None,
            repetition_penalty=None,
            bad_words_ids=None,
            num_return_sequences=None,
            decoder_start_token_id=None,
            n_docs=None,
            prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]] = None,
            **model_kwargs
    ):
        """
        Implements RAG token decoding.

        """

        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
        is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # set default parameters
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups

        max_length = max_length if max_length is not None else self.config.max_length
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.generator.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.generator.eos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.generator.pad_token_id
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.config.generator.decoder_start_token_id
        )

        # retrieve docs
        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            out = self.retriever(
                input_ids,
                question_hidden_states.cpu().detach().to(torch.float32).numpy(),
                prefix=self.generator.config.prefix,
                n_docs=n_docs,
                return_tensors="pt",
            )
            context_input_ids, context_attention_mask, retrieved_doc_embeds = (
                out["context_input_ids"],
                out["context_attention_mask"],
                out["retrieved_doc_embeds"],
            )

            # set to correct device
            retrieved_doc_embeds = retrieved_doc_embeds.to(question_hidden_states)
            context_input_ids = context_input_ids.to(input_ids)
            context_attention_mask = context_attention_mask.to(input_ids)

            # compute doc_scores
            doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(
                1
            )

        assert (
                       context_input_ids.shape[0] % n_docs
               ) == 0, f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is {context_input_ids.shape[0]}."

        # batch_size
        batch_size = context_input_ids.shape[0] // n_docs

        encoder = self.rag.generator.get_encoder()
        encoder_outputs = encoder(input_ids=context_input_ids, attention_mask=context_attention_mask)

        input_ids = torch.full(
            (batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        last_hidden_state = encoder_outputs["last_hidden_state"]

        def extend_enc_output(tensor, num_beams=None):
            # split into `batch_size`, `num_beams`, `num_docs`
            tensor = tensor[None, None, :].reshape((batch_size, 1, n_docs) + tensor.shape[1:])
            # repeat same last hidden states over `num_beams` dimension
            tensor = tensor.expand((batch_size, num_beams, n_docs) + tensor.shape[3:])
            # merge `batch_size`, `num_beams`, `num_docs` dims again
            return tensor.reshape((batch_size * num_beams * n_docs,) + tensor.shape[3:])

        # correctly extend last_hidden_state and attention mask
        context_attention_mask = extend_enc_output(context_attention_mask, num_beams=num_beams)
        encoder_outputs["last_hidden_state"] = extend_enc_output(last_hidden_state, num_beams=num_beams)

        doc_scores = doc_scores.repeat_interleave(num_beams, dim=0)

        # define start_len & additional parameters
        model_kwargs["doc_scores"] = doc_scores
        model_kwargs["encoder_outputs"] = encoder_outputs
        model_kwargs["attention_mask"] = context_attention_mask
        model_kwargs["n_docs"] = n_docs

        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            eos_token_id=eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups if not do_sample else None,
            diversity_penalty=diversity_penalty if not do_sample else None,
        )

        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # get probability distribution warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            if num_return_sequences == 1:
                # sample
                return self.sample(
                    input_ids,
                    logits_processor=logits_processor,
                    logits_warper=logits_warper,
                    max_length=max_length,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    **model_kwargs,
                )
            else:
                generated_list = []
                for i in range(num_return_sequences):
                    generated_list.append(torch.squeeze(
                        self.sample(
                            input_ids,
                            logits_processor=logits_processor,
                            logits_warper=logits_warper,
                            max_length=max_length,
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id,
                            **model_kwargs,
                        ))
                    )

                print(f"Generated list: {generated_list}")
                generated_tensor = pad_sequence(generated_list, batch_first=True,
                                                padding_value=self.config.generator.pad_token_id)

                print(f"Generated tensor: {generated_tensor}")
                return generated_tensor

        elif is_beam_gen_mode:

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )

            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if num_beams % num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            diverse_beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )

            return self.group_beam_search(
                input_ids,
                diverse_beam_scorer,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )

    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs
    ):
        """ This is the parent Huggingface method but overloaded to replace softmax with entmax.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # init sequence length tensors
        sequence_lengths, unfinished_sequences, cur_len = self._init_sequence_length_for_generation(
            input_ids, max_length
        )

        # auto-regressive generation
        while cur_len < max_length:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            scores = logits_processor(input_ids, next_token_logits)
            scores = logits_warper(input_ids, scores)

            # sample
            if not self.config.entmax:
                probs = F.softmax(scores, dim=-1)
            else:
                self.entmax_alpha = self.entmax_alpha.to(scores.device)
                #probs = entmax_bisect(scores, dim=-1, alpha=self.entmax_alpha)
                probs =  entmax15(scores, dim=-1, k=self.config.entmax_k)

            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            cur_len = cur_len + 1

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self._update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

        return input_ids

    def beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            **model_kwargs
    ):
        """  This is the parent Huggingface method but overloaded to replace softmax with entmax.
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        assert (
                num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            # adjust tokens for Bart, *e.g.*
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

            if not self.config.entmax:
                next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
            else:
                next_token_scores =  torch.log(entmax15(next_token_logits, dim=-1, k=self.config.entmax_k))


            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break

        decoded = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return decoded

    def beam_sample(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            **model_kwargs
    ):
        """  This is the parent Huggingface method but overloaded to replace softmax with entmax.
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            # adjust token scores (a no-op by default)
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

            if not self.config.entmax:
                next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
            else:
                next_token_scores = torch.log(entmax15(next_token_logits, dim=-1, k=self.config.entmax_k))

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            if not self.config.entmax:
                probs = F.softmax(next_token_scores, dim=-1)
            else:
                probs = entmax15(next_token_scores, dim=-1, k=self.config.entmax_k)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break

        decoded = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return decoded

    def group_beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            **model_kwargs
    ):
        """  This is the parent Huggingface method but overloaded to replace softmax with entmax.
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        device = input_ids.device

        batch_beam_size, cur_len = input_ids.shape

        assert (
                num_beams * batch_size == batch_beam_size
        ), f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while cur_len < max_length:
            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            # do one decoder step on all beams of all sentences in batch
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs, return_dict=True)

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []
                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of current group only
                next_token_logits = outputs.logits[batch_group_indices, -1, :]

                # adjust tokens for Bart, *e.g.*
                next_token_logits = self.adjust_logits_during_generation(
                    next_token_logits, cur_len=cur_len, max_length=max_length
                )

                next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * group_size, vocab_size)
                vocab_size = next_token_scores.shape[-1]

                next_token_scores = logits_processor(
                    group_input_ids, next_token_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores + beam_scores[batch_group_indices].unsqueeze(-1).expand_as(
                    next_token_scores
                )
                # reshape for beam search

                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = next_tokens // vocab_size
                next_tokens = next_tokens % vocab_size

                # stateless
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                        num_beams * (beam_idx // group_size) + group_start_idx + (beam_idx % group_size)
                )

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], reordering_indices)

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1
            if beam_scorer.is_done:
                break

        decoded = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return decoded

    def get_input_embeddings(self):
        return self.rag.generator.get_input_embeddings()

    def get_output_embeddings(self):
        return self.rag.generator.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.rag.generator.set_output_embeddings(new_embeddings)

    def shift_tokens_right(self, input_ids, start_token_id=None):
        """Shift input ids one token to the right, and pad with start_token_id"""
        if start_token_id is None:
            start_token_id = self.config.decoder_start_token_id
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = start_token_id
        return shifted_input_ids


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
from collections import defaultdict
from typing import Optional

import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, \
    PretrainedConfig, PreTrainedModel, RagConfig
from transformers.models.rag.modeling_rag import RetrievAugLMOutput, RagModel, RagTokenForGeneration, \
    RetrievAugLMMarginOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


def div(x, y):
    if y == 0:
        return x
    else:
        return x / y


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

        def freeze_part(model: torch.nn.Module):
            for par in model.parameters():
                par.requires_grad = False

        self.use_dataset_retrieval = config.use_dataset_retrieval
        self.use_memory_retrieval = config.use_memory_retrieval

        if self.use_memory_retrieval:
            self.context_encoder = DPRContextEncoder.from_pretrained(config.context_encoder)
            freeze_part(self.context_encoder)

            self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(config.context_encoder)
        else:
            self.context_encoder = None
            self.ctx_tokenizer = None

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
        )
        # encoder_outputs are pre-computed during RAG-token generation
        if encoder_outputs is None:

            if has_to_retrieve:

                if self.use_dataset_retrieval or self.use_memory_retrieval:
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
                    context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids = (
                        retriever_outputs["context_input_ids"],
                        retriever_outputs["context_attention_mask"],
                        retriever_outputs["retrieved_doc_embeds"],
                        retriever_outputs["doc_ids"],
                    )

                    # set to correct device
                    retrieved_doc_embeds = retrieved_doc_embeds.to(question_encoder_last_hidden_state)
                    context_input_ids = context_input_ids.to(input_ids)
                    context_attention_mask = context_attention_mask.to(input_ids)

                    # compute doc_scores
                    doc_scores = torch.bmm(
                        question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                    ).squeeze(1)

                if self.use_memory_retrieval:
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

        assert (
                doc_scores is not None
        ), "Make sure that `doc_scores` are passed when passing `encoder_outputs` to the forward function."

        assert (
                       doc_scores.shape[1] % n_docs
               ) == 0, f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is {context_input_ids.shape[0]}."

        # Decoder input without context documents
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.repeat_interleave(n_docs, dim=0)

        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.repeat_interleave(n_docs, dim=0)

        gen_outputs = self.generator(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
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
        )

    def add_to_memory(self, input_ids, attention_mask, input_text_metadata):
        with torch.no_grad():
            ctx_enc_outputs = self.context_encoder(
                input_ids, attention_mask=attention_mask, return_dict=True
            )
            # logger.info(f"Context Encoded {ctx_enc_outputs}")
            context_embeddings = ctx_enc_outputs.pooler_output.detach().cpu().to(torch.float32).numpy()
            # logger.info(f"{context_embeddings}")

            self.retriever.add(context_dicts=input_text_metadata, context_hidden_states=context_embeddings)


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

        loss = None
        logits = outputs.logits
        doc_scores = outputs.doc_scores
        context_input_ids = outputs.context_input_ids
        if labels is not None:
            assert decoder_input_ids is not None

            rag_logprobs = self.marginalize(logits, doc_scores, n_docs)

            # #print(f"nll input: {outputs.logits.size()}, {outputs.doc_scores.size()}, {labels.size()} ")
            loss = self.get_nll(
                rag_logprobs,
                doc_scores,
                labels,
                reduce_loss=reduce_loss,
                epsilon=self.config.label_smoothing,
                n_docs=n_docs,
            )

            if (torch.rand(1).item() >= self.config.unlikelihood_ratio):
                unlikelihood_loss = self.get_unlikelihood_loss(
                    rag_logprobs,
                    context_input_ids=context_input_ids,
                    unlikelihood_beta=self.config.unlikelihood_beta
                )
                #print(f"Unlikelihood loss: {unlikelihood_loss}")
                loss += unlikelihood_loss

        if do_marginalize:
            logits = self.marginalize(logits, outputs.doc_scores, n_docs)

        return RetrievAugLMMarginOutput(
            loss=loss,
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
        )

    def get_nll(self, rag_logprobs, doc_scores, target, reduce_loss=False, epsilon=0.0, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        # shift tokens left
        target = torch.cat(
            [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
        )

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        target = target.unsqueeze(-1)
        assert target.dim() == rag_logprobs.dim()

        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits
        ll, smooth_obj = _mask_pads(ll, smooth_obj)
        ll = ll.sum(1)  # sum over tokens
        # smooth_obj = smooth_obj.sum(1)

        nll_loss = -ll
        # smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            # smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss  # + eps_i * smooth_loss
        return loss

    def count_n_grams(self, tokens, n):
        n_grams = defaultdict(int)
        for n_gram in NGramIterator(tokens, n):
            n_grams[n_gram] += 1
        return n_grams

    def get_unlikelihood_loss(self, rag_logprobs, context_input_ids, unlikelihood_ngrams: int = 4,
                              unlikelihood_beta: float = 0.5):

        pred_tokens = torch.max(rag_logprobs, dim=-1)[1]

        ##print(f"Unlikelihood training: {rag_logprobs.size()}, {pred_tokens.size()}, {context_input_ids.size()}")

        crep_mask = torch.zeros_like(pred_tokens).type_as(rag_logprobs)
        lrep_mask = torch.zeros_like(pred_tokens).type_as(rag_logprobs)

        #print(f"Unlikelihood: {pred_tokens.size()}, {rag_logprobs.size()}, {context_input_ids}")
        for i, (tokens, logprob, context) in enumerate(zip(pred_tokens, rag_logprobs, context_input_ids)):
            context_ids_list = context.cpu().detach().tolist()
            context_n_grams = self.count_n_grams(context_ids_list, n=unlikelihood_ngrams)

            ##print(f"Ngrams: {context_ngrams}")

            seen_n_grams = defaultdict(int)

            # penalize if there is a context repeat
            tokens_id_list = tokens.cpu().tolist()
            for j, n_gram in enumerate(NGramIterator(tokens_id_list , unlikelihood_ngrams)):
                if context_n_grams[n_gram] > 0:
                    #print(f"Context seen: {n_gram}")
                    crep_mask[i, j: j + unlikelihood_ngrams] = 1

            for j, n_gram in enumerate(NGramIterator(tokens_id_list , unlikelihood_ngrams)):
                if seen_n_grams[n_gram] > 0:
                    #print(f"Label seen: {n_gram}")
                    lrep_mask[i, j: j + unlikelihood_ngrams] = 1
                seen_n_grams[n_gram] += 1

            #print(f"Context Ngrams: {context_n_grams}")
            #print(f"Seen Ngrams: {seen_n_grams}")

        pred_lprobs = rag_logprobs.view(-1, rag_logprobs.size(2)).gather(1, pred_tokens.view(-1, 1).long())
        #print(f"pred_lprobs: {rag_logprobs}, {pred_tokens}")

        one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=1e-6).view(
            pred_tokens.size(0), pred_tokens.size(1)
        )

        #print(f"Masks sum: {torch.sum(lrep_mask, dim=-1)},  {torch.sum(crep_mask, dim=-1)}")
        mask = ((1 - unlikelihood_beta) * lrep_mask) + (
                unlikelihood_beta * crep_mask
        )

        ul_loss = -(torch.log(one_minus_probs)) * mask
        #print(f"ul loss: {ul_loss}")
        total_loss = ul_loss.sum()#div(ul_loss.sum(), mask.sum())

        return total_loss

    def marginalize(self, seq_logits, doc_scores, n_docs=None):

        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # RAG-token marginalization
        seq_logprobs = torch.nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )
        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        return torch.logsumexp(log_prob_sum, dim=1)

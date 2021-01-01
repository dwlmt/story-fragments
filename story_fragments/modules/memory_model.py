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
from typing import Optional, Callable, List

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, \
    PretrainedConfig, PreTrainedModel, RagConfig, BeamSearchScorer
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
                    rag_logprobs=rag_logprobs,
                    context_input_ids=context_input_ids,
                    labels=labels,
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

    def get_unlikelihood_loss(self, rag_logprobs, context_input_ids, labels, unlikelihood_ngrams: int = 4,
                              unlikelihood_beta: float = 0.5):

        pred_tokens = torch.max(rag_logprobs, dim=-1)[1]

        ##print(f"Unlikelihood training: {rag_logprobs.size()}, {pred_tokens.size()}, {context_input_ids.size()}")

        crep_mask = torch.zeros_like(pred_tokens).type_as(rag_logprobs)
        lrep_mask = torch.zeros_like(pred_tokens).type_as(rag_logprobs)

        #print(f"Unlikelihood: {pred_tokens.size()}, {rag_logprobs.size()}, {context_input_ids}")
        for i, (tokens, logprob, context, lab) in enumerate(zip(pred_tokens, rag_logprobs, context_input_ids, labels)):
            context_ids_list = context.cpu().detach().tolist()
            labels_ids_list = lab.cpu().detach().tolist()

            context_n_grams = self.count_n_grams(context_ids_list, n=unlikelihood_ngrams)

            ##print(f"Ngrams: {context_ngrams}")

            seen_n_grams = defaultdict(int)

            # penalize if there is a context repeat
            tokens_id_list = tokens.cpu().detach().tolist()
            for j, n_gram in enumerate(NGramIterator(tokens_id_list , unlikelihood_ngrams)):
                if context_n_grams[n_gram] > 0 and n_gram != tuple(labels_ids_list[j: j + unlikelihood_ngrams]):
                    #print(f"Context seen: {n_gram}")
                    crep_mask[i, j: j + unlikelihood_ngrams] = 1

            for j, n_gram in enumerate(NGramIterator(tokens_id_list , unlikelihood_ngrams)):
                if seen_n_grams[n_gram] > 0 and n_gram != tuple(labels_ids_list[j: j + unlikelihood_ngrams]):
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
        #print(f"Seq Logits: {seq_logits.size()}")

        # RAG-token marginalization
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
            do_sample = None,
            early_stopping = None,
            num_beams = None,
            temperature = None,
            top_k = None,
            top_p = None,
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
                generated_tensor = pad_sequence(generated_list, batch_first=True, padding_value=PAD_VALUE)

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
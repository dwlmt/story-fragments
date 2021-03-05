import logging
from collections import deque
from typing import Dict, Any, List, Union

import numpy
import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.training.metrics import Perplexity, CategoricalAccuracy
from transformers import AutoTokenizer

from story_fragments.modules.memory_model import RagMemoryTokenForGeneration
from story_fragments.modules.memory_rag_config import RagMemoryConfig
from story_fragments.modules.memory_retriever import RagMemoryRetriever

PAD_TOKEN = 1

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


@Model.register('rag-fragments')
class RagFragmentsModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 retriever_tokenizer_name="facebook/dpr-question_encoder-multiset-base",
                 tokenizer_name: str = "facebook/bart-base",
                 question_encoder_name: str = "facebook/dpr-question_encoder-multiset-base",
                 retriever_name: str = "facebook/rag-token-base",
                 generator_name: str = "facebook/bart-base",
                 context_encoder="facebook/dpr-ctx_encoder-multiset-base",
                 ndocs: int = 5,
                 retrieval_batch_size: int = 32,
                 max_combined_length: int = 512,
                 index_name: str = "custom",
                 use_dummy_dataset: bool = False,
                 passages_path: str = None,
                 index_path: str = None,
                 dataset="wiki_dpr",
                 lm_accuracy_top_k: List[int] = [1, 5, 20],
                 gradient_checkpointing: bool = True,
                 rotate_grad_training: bool = False,
                 use_dataset_retrieval: bool = True,
                 use_memory_retrieval: bool = True,
                 memory_n_docs: int = 5,
                 memory_capacity: int = 127000,
                 memory_buffer=1000,
                 memory_lru: bool = True,
                 combined_n_docs: int = 5,
                 rag_text_concat_first: bool = False,
                 entmax: bool = False,
                 entmax_k: int = 512,
                 unlikelihood_ratio: float = 1.0,
                 unlikelihood_beta: float = 0.5,
                 train_context_encoder: bool = False,
                 ):
        super().__init__(vocab)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_tokenizer_name)

        config = RagMemoryConfig.from_pretrained(retriever_name)

        config.index_name = index_name
        config.passages_path = passages_path
        config.index_path = index_path
        config.dataset = dataset
        config.use_dummy_dataset = use_dummy_dataset
        config.n_docs = ndocs
        config.max_combined_length = max_combined_length
        config.gradient_checkpointing = gradient_checkpointing

        config.retrieval_batch_size = retrieval_batch_size

        config.use_dataset_retrieval = use_dataset_retrieval
        config.use_memory_retrieval = use_memory_retrieval
        config.memory_n_docs = memory_n_docs
        config.memory_capacity = memory_capacity
        config.memory_buffer = memory_buffer
        config.memory_lru = memory_lru
        config.combined_n_docs = combined_n_docs
        config.context_encoder = context_encoder

        config.rag_text_concat_first = rag_text_concat_first

        config.entmax = entmax
        config.entmax_k = entmax_k

        config.unlikelihood_beta = unlikelihood_beta
        config.unlikelihood_ratio = unlikelihood_ratio

        config.train_context_encoder = train_context_encoder


        self.retriever = RagMemoryRetriever.from_pretrained(retriever_name,
                                                            config=config)

        self.generator_name = generator_name

        self.model = RagMemoryTokenForGeneration.from_pretrained_question_encoder_generator(question_encoder_name,
                                                                                            generator_name,
                                                                                            config=config,
                                                                                            retriever=self.retriever)

        self.rag_ndocs = combined_n_docs

        self.lm_accuracy_top_k = lm_accuracy_top_k

        self.memory_id = 0

        '''
        self.metrics = {}

        for acc in self.lm_accuracy_top_k:
            self.metrics[f'lm_accuracy_{acc}'] = CategoricalAccuracy(top_k=acc)
        self.metrics['lm_perplexity'] = Perplexity()
        '''

    # Note that the signature of forward() needs to match that of field names
    def forward(self,
                text: TextFieldTensors,
                labels: TextFieldTensors = None,
                negative_labels: TextFieldTensors = None,
                metadata: List[Dict[str, Any]] = None,
                dataset: List[str] = None,
                ) -> Dict[str, torch.Tensor]:

        logger.debug(f"Input: {metadata}")

        results = {}

        input_ids = text["tokens"]['token_ids']
        input_mask = text["tokens"]['mask']

        batch_size = len(input_ids)

        input_text_list = []

        for id, source_text in zip([m['id'] for m in metadata], [m['text'] for m in metadata]):
            input_text_dict = {}
            input_text_dict["id"] = id
            input_text_dict["text"] = source_text
            input_text_dict["title"] = ""

            input_text_list.append(input_text_dict)

        if labels is None:

            label_tokens = None
            label_mask = None

        else:
            label_tokens = labels["tokens"]['token_ids']

        # print(f"Model Inputs: {input_ids.size()}, {label_tokens.size()}")

        # Allow n docs to be set dynamically.

        if "ndocs" in metadata[0]:
            rag_ndocs = metadata[0]["ndocs"]
        else:
            rag_ndocs = self.rag_ndocs

        if rag_ndocs > 0:
            model_output = self.model(input_ids=input_ids,
                                      attention_mask=input_mask,
                                      input_text_metadata=input_text_list,
                                      labels=label_tokens,
                                      output_retrieved=True,
                                      output_hidden_states=True,
                                      n_docs = rag_ndocs
                                      #output_attentions=True,
                                      )
        else:
            doc_scores = torch.ones(input_ids.size()[0],1).to(input_ids.device)

            model_output = self.model(context_input_ids=input_ids,
                                      context_attention_mask=input_mask,
                                      doc_scores=doc_scores,
                                      input_text_metadata=input_text_list,
                                      labels=label_tokens,
                                      output_retrieved=True,
                                      output_hidden_states=True,
                                      n_docs=rag_ndocs
                                      # output_attentions=True,
                                      )

        loss = torch.mean(model_output.loss)
        results["loss"] = loss

        label_mask = labels["tokens"]['mask']

        if not self.training:

            with torch.no_grad():

                # self._update_metrics(model_output, label_tokens, label_mask)
                self._add_retrieval_info(model_output, label_tokens, results)

                #print(f"Output: {model_output}")

                self._process_embeddings_and_metrics(batch_size, rag_ndocs, input_mask, label_mask, model_output, results)

        results["input"] = metadata

        logger.debug(f"Results: {results}")
        return results

    def _process_embeddings_and_metrics(self, batch_size, rag_ndocs, input_mask, label_mask, model_output, results):
        actual_n_docs = max(rag_ndocs, 1)
        generator_enc_last_hidden_state = model_output.generator_enc_last_hidden_state
        if actual_n_docs > 1:
            doc_scores_softmax = torch.unsqueeze(torch.softmax(model_output.doc_scores, dim=-1), dim=2)
            doc_marginalised = doc_scores_softmax * model_output.retrieved_doc_embeds * actual_n_docs
            x = torch.mean(doc_marginalised, dim=-2)
            # y = torch.max(doc_marginalised, dim=-2).values
            results["retrieved_doc_embeddings"] = x  # torch.cat((x,y), dim=-1)
            # logger.info(f"retrieved_doc_embeddings size: {results['retrieved_doc_embeddings'].size()}")

            results["retrieved_doc_probs"] = doc_scores_softmax
            results["retrieved_doc_scores"] = model_output.doc_scores
        else:
            doc_scores = None
            doc_scores_softmax = None
        if actual_n_docs > 1:
            context_mask = model_output.context_attention_mask.bool()
        else:
            context_mask = input_mask
        # print(f"generator_enc_last_hidden_state: {generator_enc_last_hidden_state.size()}")
        if generator_enc_last_hidden_state is not None:

            if generator_enc_last_hidden_state.size()[0] != batch_size:
                generator_enc_last_hidden_state = torch.unsqueeze(generator_enc_last_hidden_state, dim=0)
                context_mask = torch.unsqueeze(context_mask, dim=0)

            if doc_scores_softmax is not None:
                while len(doc_scores_softmax.size()) < len(generator_enc_last_hidden_state.size()):
                    doc_scores_softmax = torch.unsqueeze(doc_scores_softmax, dim=-1)

            generator_indexed = generator_enc_last_hidden_state[context_mask]
            if doc_scores_softmax is not None:
                marginalised = doc_scores_softmax * generator_indexed * actual_n_docs
                x = torch.mean(torch.mean((marginalised), dim=-2), dim=-2)
                # y = torch.max(torch.max((marginalised), dim=-2).values, dim=-2).values
            else:
                marginalised = generator_indexed
                x = torch.mean((marginalised), dim=-2)

            gen_enc_emb = x  # torch.cat((x,y),dim=-1)
            if len(gen_enc_emb.size()) == 1:
                gen_enc_emb = torch.unsqueeze(gen_enc_emb, dim=0)

            # print(f"generator_enc_embeddings: {gen_enc_emb.size()}")

            if len(gen_enc_emb.size()) == 1:
                gen_enc_emb = torch.unsqueeze(gen_enc_emb, dim=0)

            results["generator_enc_embeddings"] = gen_enc_emb
            # logger.info(f"generator_enc_embeddings size: {results['generator_enc_embeddings'].size()}")
        decoder_hidden_states = model_output.generator_dec_hidden_states
        if decoder_hidden_states is not None:
            decoder_hidden_states = decoder_hidden_states[0]

            decoder_hidden_states = decoder_hidden_states.view(int(decoder_hidden_states.size()[0] / actual_n_docs),
                                                               actual_n_docs, decoder_hidden_states.size()[1], -1)

            if decoder_hidden_states.size()[0] != batch_size:
                decoder_hidden_states = torch.unsqueeze(decoder_hidden_states, dim=0)

            if doc_scores_softmax is not None:
                while len(doc_scores_softmax.size()) < len(decoder_hidden_states.size()):
                    doc_scores_softmax = torch.unsqueeze(doc_scores_softmax, dim=-1)

            # logger.info(f"Marginalised: {doc_scores_softmax.size()}, {decoder_hidden_states.size()}, {label_mask.size()}")
            # label_mask = label_mask.view(int(label_mask.size()[0] / decoder_hidden_states.size()[0] / actual_n_docs))

            if doc_scores_softmax is not None:
                marginalised = doc_scores_softmax * decoder_hidden_states[
                    torch.unsqueeze(label_mask, dim=1).repeat(1, actual_n_docs, 1)] * actual_n_docs
                x = torch.mean(torch.mean((marginalised), dim=-2), dim=-2)
                # y = torch.max(torch.max((marginalised), dim=-2).values, dim=-2).values
            else:
                marginalised = decoder_hidden_states
                x = torch.mean((marginalised), dim=-2)

            gen_dec_emb = x  # torch.cat((x,y),dim=-1)
            if len(gen_dec_emb.size()) == 1:
                gen_dec_emb = torch.unsqueeze(gen_dec_emb, dim=0)

            # print(f"generator_dec_embeddings: {gen_dec_emb.size()}")

            if len(gen_dec_emb.size()) == 1:
                gen_dec_emb = torch.unsqueeze(gen_enc_emb, dim=0)

            results["generator_dec_embeddings"] = gen_dec_emb
            # logger.info(f"generator_dec_embeddings size: {results['generator_dec_embeddings'].size()}")
        if model_output.perplexity is not None:
            results["perplexity"] = model_output.perplexity
            results["avg_log_likelihood"] = model_output.avg_log_likelihood
        else:
            results["perplexity"] = 0.0
            results["avg_log_likelihood"] = 0.0

    def _pad(self, tensor, pad_to=1024, zeros=False, type=torch.long):
        if zeros:
            tensor_new = torch.zeros(tensor.size()[0], pad_to, device=tensor.device, dtype=type)
        else:
            tensor_new = torch.ones(tensor.size()[0], pad_to, device=tensor.device, dtype=type)
        tensor_new[:, 0:tensor.size()[1]] = tensor
        tensor = tensor_new
        return tensor

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "generated_sequences" in output_dict:
            print(output_dict["generated_sequences"])
            output_dict["generated_sequences"] = self.tokenizer.decode(output_dict["generated_sequences"],
                                                                       skip_special_tokens=True)
        return output_dict

    '''
    def _update_metrics(self, model_output, label_tokens, label_mask):
        if not self.training:
            with torch.no_grad():

                self.metrics['lm_perplexity'](model_output.perplexity)

                num_docs = max(self.rag_ndocs, 1)
                labels_batch_size = label_tokens.size()[0]
                indices = range(0, labels_batch_size * num_docs, num_docs)

                logits_indexed = model_output.logits[indices]
                for acc in self.lm_accuracy_top_k:
                    self.metrics[f'lm_accuracy_{acc}'](logits_indexed, label_tokens, mask=label_mask)
    '''

    def _add_retrieval_info(self, model_outputs, label_tokens, results):
        if not self.training:
            with torch.no_grad():

                num_docs = max(self.rag_ndocs, 1)

                labels_batch_size = label_tokens.size()[0]
                indices = range(0, labels_batch_size * num_docs, num_docs)

                logits_indexed = model_outputs.logits[indices]
                logits_max = torch.argmax(logits_indexed, dim=-1)

                results["predicted_tokens"] = logits_max
                results["predicted_text"] = self.tokenizer.batch_decode(logits_max.cpu().tolist(),
                                                                        skip_special_tokens=True,
                                                                        clean_up_tokenization_spaces=True)

                batch_docs_list = []

                if model_outputs.doc_scores is not None and model_outputs.retrieved_doc_ids is not None:
                    batch_dl_scores = model_outputs.doc_scores.tolist()
                    batch_doc_ids = model_outputs.retrieved_doc_ids.tolist()

                    for doc_ids, dl_scores in zip(batch_doc_ids, batch_dl_scores):

                        dp_scores = model_outputs.doc_scores.softmax(dim=-1)[0]

                        # print(f"Retrieved doc ids {model_outputs.retrieved_doc_ids}")

                        doc_dicts = []
                        print(f"BATCH_DOC_IDS: {batch_doc_ids}")
                        for doc_id in doc_ids:
                            if int(doc_id) < int(1e9):
                                doc_dict = self.retriever.index.get_doc_dicts(numpy.array([doc_id]))[0]
                            else:
                                doc_dict = self.retriever.memory_index.get_doc_dict(doc_id)

                            doc_dicts.append(doc_dict)

                        doc_list = []

                        for doc_id, dl_score, dp_score, doc_dict in zip(doc_ids, dl_scores, dp_scores,
                                                                           doc_dicts):
                            ret_doc_dict = {}
                            ret_doc_dict["id"] = doc_id
                            ret_doc_dict["title"] = doc_dict["title"]
                            ret_doc_dict["text"] = doc_dict["text"]
                            ret_doc_dict["dot_score"] = dl_score
                            ret_doc_dict["probability"] = dp_score.item()

                            doc_list.append(ret_doc_dict)

                        batch_docs_list.append(doc_list)

                    results["retrieved_docs"] = batch_docs_list

    '''
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

        for k, v in metrics.items():

            if isinstance(v, Dict):
                metrics = {**metrics, **v}

        return metrics
    '''

    def generate(self,
                 text: Union[str, List],
                 max_length: int = 128,
                 min_length: int = 10,
                 repetition_penalty: float = 0.0,
                 num_return_sequences: int = 10,
                 no_repeat_ngram_size: int = 4,
                 num_beams: int = 10,
                 num_beam_groups: int = 10,
                 length_penalty: float = 1.0,
                 diversity_penalty: float = 0.0,
                 do_deduplication=True,
                 do_sample=True,
                 temperature=1.0,
                 top_k=50,
                 top_p=0.9,
                 max_input_length: int = 512
                 ) -> List[Dict[str, Any]]:

        input_dict = self.retriever_tokenizer.encode_plus(text, max_length=max_input_length)
       
        input_ids = input_dict["input_ids"]
        attention_mask = input_dict["attention_mask"]

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        if len(input_ids.size()) == 1:
            input_ids = torch.unsqueeze(input_ids, dim=0)
            attention_mask = torch.unsqueeze(attention_mask, dim=0)

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        generated = self.model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        do_deduplication=do_deduplication,
                                        num_beams=num_beams,
                                        num_beam_groups=num_beam_groups,
                                        num_return_sequences=num_return_sequences,
                                        min_length=min_length,
                                        max_length=max_length,
                                        repetition_penalty=repetition_penalty,
                                        no_repeat_ngram_size=no_repeat_ngram_size,
                                        diversity_penalty=diversity_penalty,
                                        length_penalty=length_penalty,
                                        do_sample=do_sample,
                                        temperature=temperature,
                                        top_k=top_k,
                                        top_p=top_p,
                                        )

        generated_strings = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        generated_list = []
        for i, (gen_ids, gen_string) in enumerate(zip(generated, generated_strings)):
            generated = {"id": i, "text": gen_string}
            generated_list.append(generated)

        return generated_list

    def add_to_memory(self, text: Union[str, List],
                      add_to_memory: bool = True,
                      max_input_length: int = 512
                      ) -> Dict[str, Any]:

        if add_to_memory:
            input_dict = self.retriever_tokenizer.encode_plus(text, max_length=max_input_length)

            input_ids = input_dict["input_ids"]
            attention_mask = input_dict["attention_mask"]

            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            if len(input_ids.size()) == 1:
                input_ids = torch.unsqueeze(input_ids, dim=0)
                attention_mask = torch.unsqueeze(attention_mask, dim=0)

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()

                if isinstance(text, str):
                    text = [text]

                input_text_list = []

                for source_text in text:
                    input_text_dict = {}
                    input_text_dict["text"] = source_text
                    input_text_dict["id"] = f"{self.memory_id}"
                    input_text_dict["title"] = ""

                    input_text_list.append(input_text_dict)

                    self.memory_id += 1

                # print(f"Add to memory {input_ids}, {attention_mask}, {input_text_list}")

                self.model.rag.add_to_memory(input_ids, attention_mask, input_text_list)

    def clear_memory(self):

        self.model.rag.clear_memory()

import logging
from collections import deque
from typing import Dict, Any, List, Union

import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.training.metrics import Perplexity, CategoricalAccuracy
from transformers import AutoTokenizer

from story_fragments.models.utils import freeze_part, unfreeze_part
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
                 entmax: bool = False,
                 entmax_k: int = 512,
                 unlikelihood_ratio: float = 1.0,
                 unlikelihood_beta: float = 0.5,
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

        config.entmax = entmax
        config.entmax_k = entmax_k

        config.unlikelihood_beta = unlikelihood_beta
        config.unlikelihood_ratio = unlikelihood_ratio

        self.retriever = RagMemoryRetriever.from_pretrained(retriever_name,
                                                            config=config)

        self.generator_name = generator_name

        self.model = RagMemoryTokenForGeneration.from_pretrained_question_encoder_generator(question_encoder_name,
                                                                                            generator_name,
                                                                                            config=config,
                                                                                            retriever=self.retriever)

        self.rag_ndocs = ndocs

        self.rotate_grad_training = rotate_grad_training
        self.rotate_grad_parts_list = deque(["question_encoder", "encoder", "decoder"])

        self.lm_accuracy_top_k = lm_accuracy_top_k
        self.metrics = {}

        for acc in self.lm_accuracy_top_k:
            self.metrics[f'lm_accuracy_{acc}'] = CategoricalAccuracy(top_k=acc)
        self.metrics['lm_perplexity'] = Perplexity()

        self.memory_id = 0

    # Note that the signature of forward() needs to match that of field names
    def forward(self,
                text: TextFieldTensors,
                labels: TextFieldTensors = None,
                negative_labels: TextFieldTensors = None,
                metadata: List[Dict[str, Any]] = None,
                dataset: List[str] = None,
                num_sequences_to_generate: int = 0,
                inference: bool = False
                ) -> Dict[str, torch.Tensor]:

        logger.debug(f"Input: {metadata}")

        self._freeze_params_if_required()

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

            model_output = self.model(input_ids=input_ids,
                                      attention_mask=input_mask,
                                      input_text_metadata=input_text_list,
                                      output_retrieved=True,
                                      )

        else:

            label_tokens = labels["tokens"]['token_ids']

            # print(f"Model Inputs: {input_ids.size()}, {label_tokens.size()}")

            model_output = self.model(input_ids=input_ids,
                                      attention_mask=input_mask,
                                      input_text_metadata=input_text_list,
                                      labels=label_tokens,
                                      output_retrieved=True,
                                      )

            loss = torch.mean(model_output.loss)
            results["loss"] = loss

            if not self.training:
                label_mask = labels["tokens"]['mask']
                self._update_metrics(model_output, label_tokens, label_mask)
                self._add_retrieval_info(model_output, label_tokens, results)

        if self.training and self.rotate_grad_training:
            self.rotate_grad_parts_list.rotate(-1)

        if not self.training:

            with torch.no_grad():

                doc_scores_softmax = torch.unsqueeze(torch.softmax(model_output.doc_scores, dim=-1), dim=2)

                x = torch.mean(doc_scores_softmax * model_output.retrieved_doc_embeds * self.rag_ndocs, dim=-2)
                results["retrieved_doc_embeddings"] = x

                generator_enc_last_hidden_state = model_output.generator_enc_last_hidden_state

                context_mask = model_output.context_attention_mask.bool()

                if generator_enc_last_hidden_state.size()[0] != batch_size:
                    generator_enc_last_hidden_state = torch.unsqueeze(generator_enc_last_hidden_state, dim=0)
                    context_mask = torch.unsqueeze(context_mask, dim=0)

                while len(doc_scores_softmax.size()) < len(generator_enc_last_hidden_state.size()):
                    doc_scores_softmax = torch.unsqueeze(doc_scores_softmax, dim=-1)

                generator_indexed = generator_enc_last_hidden_state[context_mask]
                x = torch.mean((doc_scores_softmax * generator_indexed * self.rag_ndocs), dim=-2)
                x = torch.mean(x, dim=-2)
                results["generator_enc_embeddings"] = x

                results["perplexity"] = torch.exp(model_output.loss / generator_indexed.size()[-1])

        results["input"] = metadata

        logger.debug(f"Results: {results}")
        return results

    def _pad(self, tensor, pad_to=1024, zeros=False, type=torch.long):
        if zeros:
            tensor_new = torch.zeros(tensor.size()[0], pad_to, device=tensor.device, dtype=type)
        else:
            tensor_new = torch.ones(tensor.size()[0], pad_to, device=tensor.device, dtype=type)
        tensor_new[:, 0:tensor.size()[1]] = tensor
        tensor = tensor_new
        return tensor

    def _unfreeze_params(self):
        unfreeze_part(self.model)
        unfreeze_part(self.model.shared)
        unfreeze_part(self.model.question_encoder)
        unfreeze_part(self.model.generator.context_model.encoder)
        unfreeze_part(self.model.generator.context_model.decoder)

    def _freeze_params_if_required(self):
        if self.training and self.rotate_grad_training:
            freeze_part(self.model.question_encoder)
            freeze_part(self.model.generator.context_model.encoder)
            freeze_part(self.model.generator.context_model.decoder)

            part_to_unfreeze = self.rotate_grad_parts_list[0]
            if part_to_unfreeze == "question_encoder":
                unfreeze_part(self.model.question_encoder)
            elif part_to_unfreeze == "encoder":
                unfreeze_part(self.model.generator.context_model.encoder)
            elif part_to_unfreeze == "decoder":
                unfreeze_part(self.model.generator.context_model.decoder)

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "generated_sequences" in output_dict:
            output_dict["generated_sequences"] = self.tokenizer.decode(output_dict["generated_sequences"],
                                                                       skip_special_tokens=True)
        return output_dict

    def _update_metrics(self, model_output, label_tokens, label_mask):
        if not self.training:
            with torch.no_grad():

                self.metrics['lm_perplexity'](torch.mean(
                    model_output.loss / torch.sum(label_mask,(0,1))))

                num_docs = self.rag_ndocs
                labels_batch_size = label_tokens.size()[0]
                indices = range(0, labels_batch_size * num_docs, num_docs)

                logits_indexed = model_output.logits[indices]
                for acc in self.lm_accuracy_top_k:
                    self.metrics[f'lm_accuracy_{acc}'](logits_indexed, label_tokens, mask=label_mask)

    def _add_retrieval_info(self, model_outputs, label_tokens, results):
        if not self.training:
            with torch.no_grad():

                num_docs = self.rag_ndocs
                labels_batch_size = label_tokens.size()[0]
                indices = range(0, labels_batch_size * num_docs, num_docs)

                logits_indexed = model_outputs.logits[indices]
                logits_max = torch.argmax(logits_indexed, dim=-1)

                results["predicted_tokens"] = logits_max
                results["predicted_text"] = self.tokenizer.batch_decode(logits_max.cpu().tolist(),
                                                                        skip_special_tokens=True,
                                                                        clean_up_tokenization_spaces=True)

                docs_list = []

                batch_dl_scores = model_outputs.doc_scores.tolist()
                batch_doc_ids = model_outputs.retrieved_doc_ids.tolist()

                for doc_ids, dl_scores in zip(batch_doc_ids, batch_dl_scores):

                    dp_scores = model_outputs.doc_scores.softmax(dim=-1)[0]

                    # print(f"Retrieved doc ids {model_outputs.retrieved_doc_ids}")
                    doc_dicts = self.retriever.index.get_doc_dicts(model_outputs.retrieved_doc_ids)[0]

                    for doc_id, dl_score, dp_score, title, text in zip(doc_ids, dl_scores, dp_scores,
                                                                       doc_dicts["title"],
                                                                       doc_dicts["text"]):
                        ret_doc_dict = {}
                        ret_doc_dict["id"] = doc_id
                        ret_doc_dict["title"] = title
                        ret_doc_dict["text"] = text
                        ret_doc_dict["dot_score"] = dl_score
                        ret_doc_dict["probability"] = dp_score.item()

                        docs_list.append(ret_doc_dict)

                    results["retrieved_docs"] = docs_list

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

        for k, v in metrics.items():

            if isinstance(v, Dict):
                metrics = {**metrics, **v}

        return metrics

    def generate(self, text: Union[str, List],
                 add_to_memory: bool = True,
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

        print(f"Text: {text}")
        input_dict = self.retriever_tokenizer.encode_plus(text, max_length=max_input_length)
        print(f"{input_dict}")
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
        # print(f"Generated: {generated}")
        generated_strings = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        print(f"Generated strings: {generated},{generated_strings}")

        generated_list = []
        for i, (gen_ids, gen_string) in enumerate(zip(generated, generated_strings)):
            generated = {"id": i, "text": gen_string}
            generated_list.append(generated)

        # if add_to_memory:
        #    self.add_to_memory(text=text, add_to_memory=add_to_memory)

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

                if add_to_memory:
                    # print(f"Add to memory {input_ids}, {attention_mask}, {input_text_list}")
                    self.model.rag.add_to_memory(input_ids, attention_mask, input_text_list)

    def clear_memory(self):

        self.model.rag.clear_memory()

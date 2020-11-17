import logging
from collections import deque
from typing import Dict, Any, List

import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.training.metrics import Perplexity, CategoricalAccuracy
from story_fragments.modules.memory_model import RagMemoryTokenForGeneration
from story_fragments.modules.memory_rag_config import RagMemoryConfig

from story_fragments.modules.memory_retriever import RagMemoryRetriever
from transformers import RagTokenizer, \
    RagRetriever, RagTokenForGeneration, RagSequenceForGeneration, RagConfig

from story_fragments.models.utils import freeze_part, unfreeze_part

from longformer import LongformerEncoderDecoderForConditionalGeneration, LongformerEncoderDecoderConfig
from longformer.sliding_chunks import pad_to_window_size

PAD_TOKEN = 1

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

@Model.register('rag-fragments')
class RagFragmentsModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 tokenizer_name: str = "facebook/rag-token-nq",
                 question_encoder_name: str = "facebook/dpr-question_encoder-multiset-base",
                 retriever_name: str = "facebook/rag-token-nq",
                 generator_name: str = "facebook/bart-base",
                 long_attention_mode: str = "n2",
                 ndocs: int = 5,
                 max_combined_length: int = 300,
                 index_name: str = "exact",
                 use_dummy_dataset: bool = True,
                 passages_path: str = None,
                 index_path: str = None,
                 dataset="wiki_dpr",
                 lm_accuracy_top_k: List[int] = [1, 5, 20],
                 gradient_checkpointing: bool = True,
                 rotate_grad_training: bool = False,
                 use_dataset_retrieval: bool =True,
                 use_memory_retrieval:  bool =True,
                 memory_ndocs: int = 5,
                 memory_capacity: int = 19000,
                 memory_buffer=1000,
                 memory_lru: bool = True,
                 combined_ndocs: int = 5,
                 context_encoder="facebook/dpr-ctx_encoder-multiset-base",
                 ):
        super().__init__(vocab)

                # config = RagConfig.from_pretrained(model_name)
        self.tokenizer = RagTokenizer.from_pretrained(tokenizer_name)
        # self.question_encoder = DPRQuestionEncoder.from_pretrained(question_encoder_name)

        config = RagMemoryConfig.from_pretrained(retriever_name)

        config.index_name = index_name
        config.passages_path = passages_path
        config.index_path = index_path
        config.dataset = dataset
        config.use_dummy_dataset = use_dummy_dataset
        config.n_docs = ndocs
        config.max_combined_length = max_combined_length
        config.gradient_checkpointing = gradient_checkpointing

        config.use_dataset_retrieval = use_dataset_retrieval
        config.use_memory_retrieval = use_memory_retrieval
        config.memory_ndocs = memory_ndocs
        config.memory_capacity = memory_capacity
        config.memory_buffer = memory_buffer
        config.memory_lru = memory_lru
        config.combined_ndocs = combined_ndocs
        config.context_encoder = context_encoder

        self.retriever = RagMemoryRetriever.from_pretrained(retriever_name,
                                                      config=config)

        self.generator_name = generator_name
        if "long" in generator_name:
            from longformer import LongformerEncoderDecoderForConditionalGeneration, LongformerEncoderDecoderConfig

            long_config = LongformerEncoderDecoderConfig.from_pretrained(generator_name)
            long_config.attention_mode = long_attention_mode
            long_config.gradient_checkpointing = gradient_checkpointing
            self.long_config = long_config

            generator = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(generator_name,config=long_config)

            generator_name = None
        else:
            generator = None
            self.long_config = None

        self.model = RagMemoryTokenForGeneration.from_pretrained_question_encoder_generator(question_encoder_name,
                                                                                      generator_name,
                                                                                      config=config,
                                                                                      generator_model=generator,
                                                                                      retriever=self.retriever
                                                                                      )


        self.rag_ndocs = ndocs

        self.rotate_grad_training = rotate_grad_training
        self.rotate_grad_parts_list = deque(["question_encoder", "encoder", "decoder"])

        self.lm_accuracy_top_k = lm_accuracy_top_k
        self.metrics = {}

        for acc in self.lm_accuracy_top_k:
            self.metrics[f'lm_accuracy_{acc}'] = CategoricalAccuracy(top_k=acc)
        self.metrics['lm_perplexity'] = Perplexity()

    # Note that the signature of forward() needs to match that of field names
    def forward(self,
                text: TextFieldTensors,
                labels: TextFieldTensors = None,
                metadata: List[Dict[str, Any]] = None,
                dataset: List[str] = None,
                num_sequences_to_generate: int = 0,
                ) -> Dict[str, torch.Tensor]:

        self._freeze_params_if_required()

        results = {}

        input_ids = text["tokens"]['token_ids']
        input_mask = text["tokens"]['mask']

        if labels is not None:

            label_tokens = labels["tokens"]['token_ids']

            if "long" in self.generator_name and self.long_config.attention_mode in ('tvm','sliding_chunks'):
                input_ids, input_mask = pad_to_window_size(
                    input_ids, input_mask, self.long_config.attention_window[0], PAD_TOKEN)

            input_text_list = []

            for id, source_text in zip([m['id'] for m in metadata], [m['text'] for m in metadata]):
                input_text_dict = {}
                input_text_dict["id"] = id
                input_text_dict["text"] = source_text
                input_text_dict["title"] = ""
        
                input_text_list.append(input_text_dict)

            model_output = self.model(input_ids=input_ids,
                                      attention_mask=input_mask,
                                      input_text_metadata=input_text_list,
                                      labels=label_tokens,
                                      context_input_ids=torch.unsqueeze(input_ids,dim=1).repeat(1,5,1),
                                      context_attention_mask=torch.unsqueeze(input_mask, dim=1).repeat(1, 5, 1),
                                      output_retrieved=True,
                                      )

            loss = torch.mean(model_output.loss)

            results["loss"] = loss

            if not self.training:
                label_mask = labels["tokens"]['mask']
                self._update_metrics(model_output, label_tokens, label_mask)
                self._add_retrieval_info(model_output, results)

        self._generate_if_required(input_ids, input_mask, num_sequences_to_generate, results)

        if self.training and self.rotate_grad_training:
            self.rotate_grad_parts_list.rotate(-1)


        print(results)
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
        unfreeze_part(self.model.generator.model.encoder)
        unfreeze_part(self.model.generator.model.decoder)

    def _freeze_params_if_required(self):
        if self.training and self.rotate_grad_training:
            freeze_part(self.model.question_encoder)
            freeze_part(self.model.generator.model.encoder)
            freeze_part(self.model.generator.model.decoder)

            part_to_unfreeze = self.rotate_grad_parts_list[0]
            if part_to_unfreeze == "question_encoder":
                unfreeze_part(self.model.question_encoder)
            elif part_to_unfreeze == "encoder":
                unfreeze_part(self.model.generator.model.encoder)
            elif part_to_unfreeze == "decoder":
                unfreeze_part(self.model.generator.model.decoder)

    def _generate_if_required(self, input_ids, input_mask, num_sequences_to_generate, results):
        if not self.training and num_sequences_to_generate > 0:
            with torch.no_grad():
                # TODO: Make the generation parameters configurable.
                generated_sequences = self.model.generate(input_ids=input_ids,
                                                          attention_mask=input_mask,
                                                          num_return_sequences=num_sequences_to_generate,
                                                          do_sample=True,
                                                          min_length=4,
                                                          max_length=64,
                                                          no_repeat_ngram_size=3,
                                                          top_p=0.9)
                results["generated_sequences"] = generated_sequences

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "generated_sequences" in output_dict:
            output_dict["generated_sequences"] = self.tokenizer.decode(output_dict["generated_sequences"],
                                                                       skip_special_tokens=True)
        return output_dict

    def _update_metrics(self, model_output, label_tokens, label_mask):
        if not self.training:
            with torch.no_grad():

                self.metrics['lm_perplexity'](torch.mean(model_output.loss))

                print("Logits", model_output.logits.size(), label_tokens.size())

                num_docs = self.rag_ndocs
                labels_batch_size = label_tokens.size()[0]
                indices = range(0, labels_batch_size * num_docs, num_docs)

                logits_indexed = model_output.logits[indices]
                for acc in self.lm_accuracy_top_k:
                    print(logits_indexed.size(), label_tokens.size())
                    self.metrics[f'lm_accuracy_{acc}'](logits_indexed, label_tokens, mask=label_mask)

    def _add_retrieval_info(self, model_outputs, results):
        if not self.training:
            with torch.no_grad():

                docs_list = []

                batch_dl_scores = model_outputs.doc_scores.tolist()
                batch_doc_ids = model_outputs.retrieved_doc_ids.tolist()

                for doc_ids, dl_scores in zip(batch_doc_ids, batch_dl_scores):

                    dp_scores = model_outputs.doc_scores.softmax(dim=-1)[0]

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

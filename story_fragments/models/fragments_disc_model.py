import logging
from typing import Dict, Any, List

import more_itertools
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.util import pytorch_cos_sim
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, BertTokenizer
from transformers import DPRContextEncoder
from transformers import DPRQuestionEncoder

from story_fragments.modules.memory_rag_config import RagMemoryConfig
from story_fragments.modules.memory_retriever import RagMemoryRetriever

PAD_TOKEN = 1

logger = logging.getLogger(__name__)


@Model.register('disc-fragments')
class DiscFragmentsModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 tokenizer_name="bert-base-cased",
                 context_name: str = "bert-base-cased",
                 target_name: str = None,
                 context_max_seq_length: int = 256,
                 target_max_seq_length: int = 256,
                 retriever_name: str = "facebook/rag-token-base",
                 question_encoder: str = "facebook/dpr-ctx_encoder-multiset-base",
                 context_encoder: str = "facebook/dpr-ctx_encoder-multiset-base",
                 retriever_tokenizer_name: str = "facebook/dpr-question_encoder-multiset-base",
                 ndocs: int = 5,
                 retrieval_batch_size: int = 32,
                 max_combined_length: int = 512,
                 index_name: str = "custom",
                 use_dummy_dataset: bool = True,
                 passages_path: str = None,
                 index_path: str = None,
                 dataset="wiki_dpr",
                 use_dataset_retrieval: bool = True,
                 use_memory_retrieval: bool = True,
                 memory_n_docs: int = 5,
                 memory_capacity: int = 127000,
                 memory_buffer=1000,
                 memory_lru: bool = True,
                 combined_n_docs: int = 5,
                 gradient_checkpointing: bool = True):
        super().__init__(vocab)

        self.reference_embedding_weight = torch.tensor(0.5)
        self.reference_embedding_weight = self.reference_embedding_weight.clamp(0.499, 0.501)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.question_tokenizer = AutoTokenizer.from_pretrained(retriever_tokenizer_name)

        if retriever_name is not None:
            self.question_encoder = DPRQuestionEncoder.from_pretrained(question_encoder)
            self.context_encoder = DPRContextEncoder.from_pretrained(context_encoder)

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

            self.config = config

            self.retriever = RagMemoryRetriever.from_pretrained(retriever_name,
                                                                config=config)
        else:
            self.retriever = None

        word_context_model = models.Transformer(context_name, max_seq_length=context_max_seq_length)
        pooling_context_model = models.Pooling(word_context_model.get_word_embedding_dimension())
        # dense_context_model = models.Dense(in_features=pooling_context_model.get_sentence_embedding_dimension(), out_features=768,
        #                           activation_function=torch.nn.Tanh())
        self.context_model = SentenceTransformer(
            modules=[word_context_model, pooling_context_model])  # , dense_context_model])

        if target_name is None:
            self.target_model = self.context_model
        else:
            word_target_model = models.Transformer(target_name, max_seq_length=target_max_seq_length)
            pooling_target_model = models.Pooling(word_target_model.get_word_embedding_dimension())
            # dense_target_model = models.Dense(in_features=pooling_target_model.get_sentence_embedding_dimension(),
            #                           out_features=768,
            #                           activation_function=torch.nn.Tanh())
            self.target_model = SentenceTransformer(
                modules=[word_target_model, pooling_target_model])  # , dense_target_model])

        self.metrics = {}

    # Note that the signature of forward() needs to match that of field names
    def forward(self,
                text: TextFieldTensors,
                labels: TextFieldTensors = None,
                negative_labels: TextFieldTensors = None,
                metadata: List[Dict[str, Any]] = None,
                dataset: List[str] = None,
                num_sequences_to_generate: int = 0,
                ) -> Dict[str, torch.Tensor]:

        results = {}

        input_ids = text["tokens"]['token_ids']
        input_mask = text["tokens"]['mask']

        input_mask = (input_ids != 0)
        input_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        context_output = self.context_model(input_dict)["sentence_embedding"]

        if negative_labels is not None:
            negative_ids = negative_labels["tokens"]['token_ids']
        else:
            negative_ids = None

        if self.retriever is not None and self.config.combined_n_docs > 0:
            input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)
            # print(f"Input Text: {input_text}")
            input_questions = self.question_tokenizer.batch_encode_plus(input_text, return_tensors='pt', padding=True,
                                                                        truncation=True)
            # print(f"Input Questions: {input_questions}")
            input_question_ids = input_questions["input_ids"].to(input_ids.device)
            question_encoder_last_hidden_state = self.question_encoder(input_question_ids, return_dict=True)
            # print(f"Question Hidden States: {question_encoder_last_hidden_state}")
            question_encoder_last_hidden_state = question_encoder_last_hidden_state.pooler_output
            # print(
            # f"Question Vectors: {question_encoder_last_hidden_state}, {question_encoder_last_hidden_state.size()}")

            retriever_outputs = self.retriever.retrieve(
                question_encoder_last_hidden_state.cpu().detach().float().numpy(), self.config.combined_n_docs)

            #print(f"Retrieved docs: {retriever_outputs}")

            retrieved_doc_embeds, retrieved_doc_ids, retrieved_doc_text = retriever_outputs

            #print(f"Retreived doc text: {retrieved_doc_text}")
            extracted_doc_texts = [t['text'] for t in retrieved_doc_text]
            extracted_doc_texts_flattened = []
            for ext in extracted_doc_texts:
                extracted_doc_texts_flattened.extend(ext)


            #print(f"Extracted text: {extracted_doc_texts_flattened}")
            retrieved_encoded_text_ids = self.tokenizer.batch_encode_plus(extracted_doc_texts_flattened, return_tensors='pt',
                                                                    padding=True,
                                                                    truncation=True)

            #print(f"Encoded text ids: {retrieved_encoded_text_ids}")
            encoded_retrieved_contexts = self.context_model({"input_ids": retrieved_encoded_text_ids["input_ids"].to(question_encoder_last_hidden_state.device),
                                                 "attention_mask": retrieved_encoded_text_ids["attention_mask"].to(question_encoder_last_hidden_state.device)})[
                "sentence_embedding"]

            #encoded_retrieved_contexts = torch.stack(encoded_retrieved_contexts_list)
            encoded_retrieved_contexts = encoded_retrieved_contexts.view(int(encoded_retrieved_contexts.size()[0] / self.config.combined_n_docs), self.config.combined_n_docs, -1)

            # set to correct device
            retrieved_doc_embeds = torch.tensor(retrieved_doc_embeds).to(
                device=question_encoder_last_hidden_state.device).to(dtype=question_encoder_last_hidden_state.dtype)

            # compute doc_scores
            doc_scores = torch.bmm(
                question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
            ).squeeze(1)

            # print(f"Doc Scores: {doc_scores}, {doc_scores.size()}")

            doc_probs = F.softmax(doc_scores, dim=-1)
            doc_probs_exp = (torch.unsqueeze(doc_probs, dim=2)).expand_as(encoded_retrieved_contexts)
            #print(f"Pre agg: {doc_probs_exp}, {doc_probs_exp.size()}, {retrieved_doc_embeds}, {retrieved_doc_embeds.size()}")

            agg_vector = torch.sum((doc_probs_exp * encoded_retrieved_contexts), dim=-2)
            #print(f"Agg vector: {context_output}, {context_output.size()},{agg_vector}, {agg_vector.size()}")

            context_output = (context_output * (1.0 - self.reference_embedding_weight)) + (
                    agg_vector * self.reference_embedding_weight)

            with torch.no_grad():

                input_text_list = []

                for id, source_text in zip([m['id'] for m in metadata], [m['text'] for m in metadata]):
                    input_text_dict = {}
                    input_text_dict["id"] = id
                    input_text_dict["text"] = source_text
                    input_text_dict["title"] = ""

                    input_text_list.append(input_text_dict)

                ctx_enc_outputs = self.context_encoder(input_question_ids, return_dict=True
                                                       )
                # logger.info(f"Context Encoded {ctx_enc_outputs}")
                context_embeddings = ctx_enc_outputs.pooler_output.detach().cpu().to(torch.float32).numpy()
                # logger.info(f"{context_embeddings}")

                self.retriever.add(context_dicts=input_text_list, context_hidden_states=context_embeddings)

        if labels is not None:

            label_ids = labels["tokens"]['token_ids']
            label_mask = (label_ids != 0)
            label_dict = {"input_ids": label_ids, "attention_mask": label_mask}
            label_output = self.target_model(label_dict)["sentence_embedding"]

            if negative_ids is not None:
                neg_label_mask = (negative_ids != 0)
                neg_label_dict = {"input_ids": negative_ids, "attention_mask": neg_label_mask}
                neg_label_output = self.target_model(neg_label_dict)["sentence_embedding"]
            else:
                neg_label_output = None

            #print(f"Label ids: {label_ids}, Label mask : {label_mask}, Negative ids: {negative_ids}, "
            #      f"Negative Mask: {neg_label_mask}, Context Output: {context_output}")

            if neg_label_output is not None:
                torch.cat((label_output, neg_label_output))
            scores = torch.mm(context_output, label_output.transpose(0, 1))
            #  pytorch_cos_sim(context_output,label_output) * 20

            scores[torch.isnan(scores)] = 0.0

            labels = torch.tensor(range(len(scores)), dtype=torch.long,
                                  device=scores.device)
            # #print(f"Labels: {labels}")

            self.cross_entropy_loss = CrossEntropyLoss()
            loss = self.cross_entropy_loss(scores, labels)
            # #print(f"Loss {loss}")

            results["loss"] = loss

        return results

    def _update_metrics(self, model_output, label_tokens):
        with torch.no_grad():
            # Only update metrics when not training to improve performance
            if not self.training:

                logits = model_output[1]

                mask = (label_tokens != PAD_TOKEN)

                for acc in self.lm_accuracy_top_k:
                    # #print(logits.size(), label_tokens.size())
                    self.metrics[f'lm_accuracy_{acc}'](logits, label_tokens, mask=mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

        for k, v in metrics.items():

            if isinstance(v, Dict):
                metrics = {**metrics, **v}

        return metrics

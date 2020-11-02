from typing import Dict, Any, List

import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.training.metrics import Perplexity, CategoricalAccuracy
from transformers import RagTokenizer, \
    RagRetriever, RagTokenForGeneration

PAD_TOKEN = 1


@Model.register('rag-fragments')
class RagFragmentsModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 tokenizer_name: str = "facebook/rag-token-nq",
                 question_encoder_name: str = "facebook/dpr-question_encoder-multiset-base",
                 retriever_name: str = "facebook/rag-token-nq",
                 generator_name: str = "facebook/bart-base",
                 rag_ndocs: int = 5,
                 rag_index_name: str = "exact",
                 rag_use_dummy_dataset: bool = True,
                 rag_passages_path: str = None,
                 rag_index_path: str = None,
                 rag_dataset="wiki_dpr",
                 lm_accuracy_top_k: List[int] = [1, 5, 20]
                 ):
        super().__init__(vocab)

        # config = RagConfig.from_pretrained(model_name)
        self.tokenizer = RagTokenizer.from_pretrained(tokenizer_name)
        # self.question_encoder = DPRQuestionEncoder.from_pretrained(question_encoder_name)

        self.retriever = RagRetriever.from_pretrained(retriever_name, index_name=rag_index_name,
                                                      passages_path=rag_passages_path,
                                                      index_path=rag_index_path,
                                                      dataset=rag_dataset,
                                                      use_dummy_dataset=rag_use_dummy_dataset,
                                                      gradient_checkpointing=True)

        # self.generator = AutoModel.from_pretrained(generator_name)

        self.model = RagTokenForGeneration.from_pretrained_question_encoder_generator(question_encoder_name,
                                                                                      generator_name,
                                                                                      retriever=self.retriever,
                                                                                      gradient_checkpointing=True,
                                                                                      )

        self.rag_ndocs = rag_ndocs

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

        print(metadata)
        results = {}

        input_ids = text["tokens"]['token_ids']
        input_mask = text["tokens"]['mask']

        if labels is not None:

            label_tokens = labels["tokens"]['token_ids']

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                label_tokens = label_tokens.cuda()

            # print(input_ids, label_tokens)
            model_output = self.model(input_ids=input_ids,
                                      attention_mask=input_mask,
                                      labels=label_tokens,
                                      output_retrieved=True,
                                      )

            #print(model_output)

            results["loss"] = torch.mean(model_output.loss)

            label_mask = labels["tokens"]['mask']

            self._update_metrics(model_output, label_tokens, label_mask)
            self._add_retrieval_info(model_output, results)

        self._generate_if_required(input_ids, input_mask, num_sequences_to_generate, results)

        print(results)
        return results

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
                indices = range(0, labels_batch_size * num_docs , num_docs)

                logits_indexed = model_output.logits[indices]
                for acc in self.lm_accuracy_top_k:
                    print(logits_indexed.size(), label_tokens.size())
                    self.metrics[f'lm_accuracy_{acc}'](logits_indexed, label_tokens, mask=label_mask)

    def _add_retrieval_info(self, model_outputs, results):
        # Only update metrics when not training to improve performance
        if not self.training:
            with torch.no_grad():

                docs_list = []

                batch_dl_scores = model_outputs.doc_scores.tolist()
                batch_doc_ids = model_outputs.retrieved_doc_ids.tolist()

                for doc_ids, dl_scores in zip(batch_doc_ids, batch_dl_scores):

                    dp_scores = model_outputs.doc_scores.softmax(dim=-1)[0]

                    doc_dicts = self.retriever.index.get_doc_dicts(model_outputs.retrieved_doc_ids)[0]


                    for doc_id, dl_score, dp_score, title, text in zip(doc_ids, dl_scores, dp_scores, doc_dicts["title"],
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

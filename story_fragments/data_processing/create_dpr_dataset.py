import csv
import os
from functools import partial
from typing import List, OrderedDict

import faiss
import fire
import more_itertools
import torch
from datasets import load_dataset
from jsonlines import jsonlines
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast

try:
    from blingfire import text_to_sentences
except ImportError:
    print("Please run `pip install blingfire`")


def embed(examples: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(examples["text"], truncation=True, padding="longest", return_tensors="pt"
                              )["input_ids"]

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    embeddings = ctx_encoder(input_ids, return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


class ProcessDPRDataset(object):

    def create(self, datasets: List[str],
               base_output_dir,
               dataset_name,
               window_size: int = 4,
               window_step=2,
               batch_size=16,
               index_name: str = "compressed",
               embedding_dim: int = 768,
               train_size: int = 250000,
               index_worlds: int = 32,
               index_ncentroids: int = 4096,
               index_code_size: int = 64,
               rag_context_encoder: str = "facebook/dpr-ctx_encoder-multiset-base",
               rag_tokenizer: str = "facebook/dpr-ctx_encoder-single-nq-base",
               skip_splitting: bool = False):

        from pathlib import Path
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)

        if not skip_splitting:
            self.split_passages_and_write_text(base_output_dir, dataset_name, datasets, window_size, window_step)

        # Reload the text dataset and process as a DPR dataset.
        dataset = load_dataset(
            "json", data_files=[f'{base_output_dir}/{dataset_name}.jsonl'], split="train")

        ctx_encoder = DPRContextEncoder.from_pretrained(rag_context_encoder)
        ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(rag_tokenizer)

        if torch.cuda.is_available():
            ctx_encoder = ctx_encoder.cuda()

        dataset = dataset.map(
            partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
            batched=True,
            batch_size=batch_size,
        )

        # Save the dataset
        passages_path = os.path.join(base_output_dir, dataset_name)
        dataset.save_to_disk(passages_path)

        if index_name == "exact":
            d = embedding_dim
            index = faiss.IndexHNSWFlat(d, index_worlds, faiss.METRIC_INNER_PRODUCT)
            dataset.add_faiss_index("embeddings", custom_index=index)
        else:
            d = embedding_dim
            quantizer = faiss.IndexHNSWFlat(d, index_worlds, faiss.METRIC_INNER_PRODUCT)
            ivf_index = faiss.IndexIVFPQ(quantizer, d, index_ncentroids,
                                         index_code_size, 8,
                                         faiss.METRIC_INNER_PRODUCT)
            ivf_index.own_fields = True
            quantizer.this.disown()
            dataset.add_faiss_index(
                "embeddings",
                custom_index=ivf_index,
                train_size=train_size,
                faiss_verbose=True
            )

        # And save the index
        index_path = os.path.join(base_output_dir, f"{dataset_name}.faiss")
        dataset.get_index("embeddings").save(index_path)

    def split_passages_and_write_text(self, base_output_dir, dataset_name, datasets, window_size, window_step):
        if isinstance(datasets, str):
            datasets = [datasets]
        id = 0
        output_list = []
        for dataset in datasets:

            with jsonlines.open(dataset) as reader:
                for obj in reader:
                    print(f"{obj['title']} - {obj['text']}")

                    sentences = text_to_sentences(f"{obj['text']}").split('\n')
                    passages = more_itertools.windowed(sentences, n=window_size, step=window_step, fillvalue=" ")
                    for i, p in enumerate(passages):
                        joined_text = " ".join(p).strip()

                        passage_dict = OrderedDict()
                        passage_dict["id"] = f"{id}"
                        passage_dict["title"] = f"{obj['id']}-{i}: {obj['title']}"
                        passage_dict["text"] = joined_text

                        #print(f"Passage: {passage_dict}")
                        output_list.append(passage_dict)

                        id += 1

        with jsonlines.open(f'{base_output_dir}/{dataset_name}.jsonl', mode='w') as writer:
            for output in output_list:
                writer.write(output)


if __name__ == '__main__':
    fire.Fire(ProcessDPRDataset)

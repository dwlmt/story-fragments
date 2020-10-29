# Processing Datasets

**create_dpr_dataset.py** is a script for creating a DPR dataset from a jsonl dataset
with *id*, *title*, and *text* source fields. The script split sentences with Blingfire, and has
sliding window options for the number of sentences and overlap in each passage. As well as the processed
text, a Huggingface dataset is saved with embeddings and a separate *.faiss index. The faiss defaults
follows those of DPR for *exact* and *compressed* options.
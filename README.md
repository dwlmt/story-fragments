# Memory and Knowledge Augmented Language Models for Inferring Salience in Long-Form Stories

This is the code repository for the following [[paper]](https://aclanthology.org/2021.emnlp-main.65/):

## Abstract

Measuring event salience is essential in the understanding of stories. This paper takes a recent unsupervised method for salience detection derived from Barthes Cardinal Functions and theories of surprise and applies it to longer narrative forms. We improve the standard transformer language model by incorporating an external knowledgebase (derived from Retrieval Augmented Generation) and adding a memory mechanism to enhance performance on longer works. We use a novel approach to derive salience annotation using chapter-aligned summaries from the Shmoop corpus for classic literary works. Our evaluation against this data demonstrates that our salience detection model improves performance over and above a non-knowledgebase and memory augmented language model, both of which are crucial to this improvement.

## Citation

```
@inproceedings{wilmot-keller-2021-memory,
    title = "Memory and Knowledge Augmented Language Models for Inferring Salience in Long-Form Stories",
    author = "Wilmot, David  and
      Keller, Frank",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.65",
    pages = "851--865"}
```

## Models

Model files and other [[resources]](https://drive.google.com/drive/folders/1RYPgdKLLIXLgVM_VlWimA332_yP5P1cZ?usp=sharing):

- **Models**: Allennlp model files from the paper and with unlikelihood training.
- **Shmoop**: Shmoop aligned summaries using Roberta Large. Note, permission is required from Shmoop to publish with this dataset.
- **WikiPlots KB**: The Wikiplots KB. Contains the source files in json format and processed with an exact faiss index.

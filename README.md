# Binary Vector Search Examples for USearch

This repository contains examples for constructing binary vector-search indicies for WikiPedia embeddings available on the HuggingFace portal:

- [Co:here](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3)
- [MixedBread.ai](https://huggingface.co/datasets/mixedbread-ai/wikipedia-embed-en-2023-11)

---

To view the results, check out the [`bench.ipynb`](bench.ipynb).
To replicate the results, first, download the data:

```sh
$ pip install -r requirements.txt
$ python download.py
$ ls -alh mixedbread | head -n 1
> total 15G
$ ls -alh cohere | head -n 1
> total 15G
```

In both cases, the embeddings have 1024 dimensions, each represented with a single bit, packed into 128-byte vectors.
32 GBs of RAM are recommended to run the scripts.

# Binary Vector Search Examples for USearch

This repository contains an example for constructing a binary vector-search index for WikiPedia embeddings by Co:here and MixedBread.

--

To view the results, check out the [`bench.ipynb`](bench.ipynb).
To replicate the results, first, download the data:

```sh
$ pip install -r requirements.tx
$ python download.py
$ ls -alh mixedbread | head -n 1
> total 15G
$ ls -alh cohere | head -n 1
> total 15G
```

In both cases, the embeddings have 1024 dimensions, each represented with a single bit, packed into 128-byte vectors.
32 GBs of RAM are recommended to run the scripts.
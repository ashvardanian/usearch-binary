# Binary Vector Search Examples for USearch

This repository contains examples for constructing binary vector-search indicies for WikiPedia embeddings available on the HuggingFace portal:

- [Co:here](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3)
- [MixedBread.ai](https://huggingface.co/datasets/mixedbread-ai/wikipedia-embed-en-2023-11)

## Running Examples

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

## Optimizations

Knowing the length of embeddings is very handy for optimizations.
If the embeddings are only 1024 bits long, we only need 2 ZMM registers to store the entire vector.
We don't need any `for`-loops, then entire operation can be unrolled and inlined.

```c
inline uint64_t hamming_distance(uint8_t const* first_vector, uint8_t const* second_vector) {
    __m512i const first_start = _mm512_loadu_si512((__m512i const*)(first_vector));
    __m512i const first_end = _mm512_loadu_si512((__m512i const*)(first_vector + 64));
    __m512i const second_start = _mm512_loadu_si512((__m512i const*)(second_vector));
    __m512i const second_end = _mm512_loadu_si512((__m512i const*)(second_vector + 64));
    __m512i const differences_start = _mm512_xor_epi64(first_start, second_start);
    __m512i const differences_end = _mm512_xor_epi64(first_end, second_end);
    __m512i const population_start = _mm512_popcnt_epi64(differences_start);
    __m512i const population_end = _mm512_popcnt_epi64(differences_end);
    __m512i const population = _mm512_add_epi64(population_start, population_end);
    return _mm512_reduce_add_epi64(population);
}
```

To run the kernel benchmarks, use the following command:

```sh
$ python kernel.py
```
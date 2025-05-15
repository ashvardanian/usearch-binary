# Optimizing Binary Vector Search with USearch

The ultimate compressed vector representation is a vector of individual bits, as opposed to more common 32-bit `f32` floats and 8-bit `i8` integers.
That representation is natively supported by [USearch](https://github.com/unum-cloud/usearch), but given the tiny size of the vectors, more optimizations can be explored to scale to larger datasets.
Luckily, modern embedding models are often trained in Quantization-aware manner, and precomputed WikiPedia embeddings are available on the HuggingFace portal:

- [Co:here](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3)
- [MixedBread.ai](https://huggingface.co/datasets/mixedbread-ai/wikipedia-embed-en-2023-11)

Native optimizations are implemented [NumBa](https://numba.pydata.org/) and [Cppyy](https://cppyy.readthedocs.io/) JIT compiler for Python and C++, and are packed into [UV](https://docs.astral.sh/uv/)-compatible scripts.

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
If the embeddings are only 1024 bits long, we only need 2 `ZMM` CPU registers on x86 machines to store the entire vector.
We don't need any `for`-loops, the entire operation can be unrolled and inlined.

```c
uint64_t hamming_distance_1024d(uint8_t const* first_vector, uint8_t const* second_vector) {
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

As shown in [less_slow.cpp](https://github.com/ashvardanian/less_slow.cpp), decomposing `for`-loops (which are equivalent to `if`-statements and jumps) into unrolled kernels is a universally great idea.
But the problem with the kernel above is that `_mm512_popcnt_epi64` is an expensive instruction, and most Intel CPUs can only execute it on a single CPU port.
There are several ways to implement population counts on SIMD-capable x86 CPUs.
Focusing on AVX-512, we can either use the `VPOPCNTQ` instruction, or the `VPSHUFB` instruction to shuffle bits and then use `VPSADBW` to sum them up:

- [VPOPCNTQ (ZMM, ZMM)](https://uops.info/html-instr/VPOPCNTQ_ZMM_ZMM.html):
    - On Ice Lake: 3 cycles latency and executes only on port 5.
    - On Zen4: 2 cycles and executes on both port 0 and 1.
- [VPSHUFB (ZMM, ZMM, ZMM)](https://uops.info/html-instr/VPSHUFB_ZMM_ZMM_ZMM.html):
    - On Skylake-X: 1 cycle latency and executes only on port 5.
    - On Ice Lake: 1 cycle latency and executes only on port 5.
    - On Zen4: 2 cycles and executes on both port 1 and 2.
- [VPSADBW (ZMM, ZMM, ZMM)](https://uops.info/html-instr/VPSADBW_ZMM_ZMM_ZMM.html)
    - On Ice Lake: 3 cycles latency and executes only on port 5.
    - On Zen4: 3 cycles and executes on both port 0 and 1.

So when implementing the Jaccard distance, the most important kernel for binary similarity indices using the `VPOPCNTQ`, we will have 4 such instruction invocations that will all stall on the same port number 5:

```c
float jaccard_distance_1024d(uint8_t const* first_vector, uint8_t const* second_vector) {
    __m512i const first_start = _mm512_loadu_si512((__m512i const*)(first_vector));
    __m512i const first_end = _mm512_loadu_si512((__m512i const*)(first_vector + 64));
    __m512i const second_start = _mm512_loadu_si512((__m512i const*)(second_vector));
    __m512i const second_end = _mm512_loadu_si512((__m512i const*)(second_vector + 64));
    __m512i const intersection_start = _mm512_and_epi64(first_start, second_start);
    __m512i const intersection_end = _mm512_and_epi64(first_end, second_end);
    __m512i const union_start = _mm512_or_epi64(first_start, second_start);
    __m512i const union_end = _mm512_or_epi64(first_end, second_end);
    __m512i const population_intersection_start = _mm512_popcnt_epi64(intersection_start);
    __m512i const population_intersection_end = _mm512_popcnt_epi64(intersection_end);
    __m512i const population_union_start = _mm512_popcnt_epi64(union_start);
    __m512i const population_union_end = _mm512_popcnt_epi64(union_end);
    __m512i const population_intersection = _mm512_add_epi64(population_intersection_start, population_intersection_end);
    __m512i const population_union = _mm512_add_epi64(population_union_start, population_union_end);
    return 1.f - _mm512_reduce_add_epi64(population_intersection) * 1.f / _mm512_reduce_add_epi64(population_union);
}
```

That's known to be a bottleneck and can be improved.

### Harley-Seal and Odd-Majority Algorithms

Let's take a look at a few 192-dimensional bit-vectors for simplicity.
Each will be represented with 3x 64-bit unsigned integers.

```cpp
std::uint64_t a[3] = {0x0000000000000000, 0x0000000000000000, 0x0000000000000000};
std::uint64_t b[3] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
std::uint64_t c[3] = {0x0000000000000000, 0xFFFFFFFFFFFFFFFF, 0x0000000000000000};
```

A naive Jaccard distance implementation in C++20 using STL's [`std::popcount`](https://en.cppreference.com/w/cpp/numeric/popcount) would look like this:

```cpp
#include <bit>

float jaccard_distance_192d(std::uint64_t const* a, std::uint64_t const* b) {
    int intersection = std::popcount(a[0] & b[0]) + std::popcount(a[1] & b[1]) + std::popcount(a[2] & b[2]);
    int union_ = std::popcount(a[0] | b[0]) + std::popcount(a[1] | b[1]) + std::popcount(a[2] | b[2]);
    return 1.f - intersection * 1.f / union_;
}
```

That's 6x `std::popcount` and we can easily reduce it to 4x, by using the following rule for both `intersection` and `union_`:

$$
\begin{aligned}
X &:= \{x_0,\,x_1,\,x_2\} \\
\operatorname{PopCount}(X) &:= \sum_{i=0}^{2}\operatorname{PopCount}(x_i) \\
\text{Odd} &:= (x_0 \oplus x_1)\,\oplus\, x_2 \\
\text{Major} &:= \bigl((x_0 \oplus x_1)\land x_2\bigr)\,\lor\, (x_0 \land x_1) \\
\curvearrowright \operatorname{PopCount}(X) &:= 2\,\operatorname{PopCount}(\text{Major}) + \operatorname{PopCount}(\text{Odd})
\end{aligned}
$$

That rule is an example of Carry-Save Adders (CSAs) and can be used for longer sequences as well.
So $N$ "Odd-Major" population counts can cover the logic needed for the $2^N-1$ original counts:

- 7x `std::popcount` can be reduced to 3x.
- 15x `std::popcount` can be reduced to 4x.
- 31x `std::popcount` can be reduced to 5x.

When dealing with 1024-dimensional bit-vectors on 64-bit machines, we can view them as 16x 64-bit words, leveraging the "Odd-Majority" trick to fold first 15x `std::popcount` into 4x, and then having 1x more call for the tail, thus shrinking from 16x to 5x calls, at the cost of several `XOR` and `AND` operations.

### Testing and Profiling Kernels

To run the kernel benchmarks, use the following command:

```sh
$ python kernel.py
```

To run benchmarks over real data:

```sh
$ python kernels.py --dir cohere --limit 1e6
```

# Jaccard Index Optimization

Jaccard Index is one of the most common tools in Information Retrieval and is used to measure the similarity between two sets, mostly defined at a single bit level:

$$
\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

In code, one would rarely deal with `boolean` values due to obvious space-inefficiency, and would generally operate on octets of bits, packed into 8-bit unsigned integers, like this:

```cpp
#include <bits> // brings `std::popcount`
#include <cstdint> // brings `std::size_t`

float jaccard(std::uint8_t const* a_vector, std::uint8_t const* b_vector, std::size_t count_octets) {
    std::size_t intersection = 0, union_ = 0;
    for (std::size_t i = 0; i < count_octets; ++i) {
        std::uint8_t a_octet = a_vector[i], b_octet = b_vector[i];
        intersection += std::popcount(a_octet & b_octet);
        union_ += std::popcount(rst_octet | b_octet);
    }
    return (float)intersection / (float)union_;
}
```

That's, however, horribly inefficient!
Assuming how often bit-level representation are now used in large-scale vector search deployments with [USearch](https://github.com/unum-cloud/usearch), this repository provides benchmarks and custom kernels exploring the performance impacts of following optimizations:

- Using lookup tables to speed up population counts.
- Using [Harley-Seal](https://en.wikipedia.org/wiki/Harley%E2%80%93Seal_adder) and Odd-Majority [CSAs](https://en.wikipedia.org/wiki/Carry-save_adder) for population counts.
- Loop unrolling and inlining.
- 🔜 Floating-point operations instead of integer ones.

---

Native optimizations are implemented [NumBa](https://numba.pydata.org/) and [Cppyy](https://cppyy.readthedocs.io/) JIT compiler for Python and C++, and are packed into [UV](https://docs.astral.sh/uv/)-compatible scripts.
For benchmarks, a combination of random and real data is used.
Luckily, modern embedding models are often trained in Quantization-aware manner, and precomputed WikiPedia embeddings are available on the HuggingFace portal:

- [Co:here](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3)
- [MixedBread.ai](https://huggingface.co/datasets/mixedbread-ai/wikipedia-embed-en-2023-11)

Most of those vectors are 1024- to 1536-dimensional, with datasets containing between 10M and 300M vectors for multilingual.

## Running Examples

To benchmark and test the kernels on your hardware, run the following command:

```sh
$ uv run --script kernels.py --count 10000 --ndims "1024,1536"
$ uv run --script kernels.py --help # for more options
```

To download and quantize some baseline data.
The following script will pull the English WikiPedia embeddings, save them to `cohere/en*`, and quantize them to 1-bit vectors:

```sh
$ uv run --script download.py --dataset cohere-en --quantize
$ uv run --script download.py --help # for more options
```

## Optimizations

Knowing the length of embeddings is very handy for optimizations.
If the embeddings are only 1024 bits long, we only need 2 `ZMM` CPU registers on x86 machines to store the entire vector.
We don't need any `for`-loops, the entire operation can be unrolled and inlined.

```c
uint64_t hamming_distance_1024d(uint8_t const* a_vector, uint8_t const* b_vector) {
    __m512i a_start = _mm512_loadu_si512((__m512i const*)(a_vector));
    __m512i a_end = _mm512_loadu_si512((__m512i const*)(a_vector + 64));
    __m512i b_start = _mm512_loadu_si512((__m512i const*)(b_vector));
    __m512i b_end = _mm512_loadu_si512((__m512i const*)(b_vector + 64));
    __m512i differences_start = _mm512_xor_epi64(a_start, b_start);
    __m512i differences_end = _mm512_xor_epi64(a_end, b_end);
    __m512i population_start = _mm512_popcnt_epi64(differences_start);
    __m512i population_end = _mm512_popcnt_epi64(differences_end);
    __m512i population = _mm512_add_epi64(population_start, population_end);
    return _mm512_reduce_add_epi64(population);
}
```

As shown in [less_slow.cpp](https://github.com/ashvardanian/less_slow.cpp), decomposing `for`-loops (which are equivalent to `if`-statements and jumps) into unrolled kernels is a universally great idea.
But the problem with the kernel above is that `_mm512_popcnt_epi64` is an expensive instruction, and most Intel CPUs can only execute it on a single CPU port.
There are several ways to implement population counts on SIMD-capable x86 CPUs, mostly relying on the following instructions:

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
- [VPDPBUSDS (ZMM, ZMM, ZMM)](https://uops.info/html-instr/VPDPBUSDS_ZMM_ZMM_ZMM.html)
    - On Ice Lake: 5 cycles latency and executes only on port 0.
    - On Zen4: 4 cycles and executes on both port 0 and 1.

Interestingly, the `EVEX` variants of `VPSHUFB` and `VPDPBUSDS` instructions take different ports when dealing with `YMM` inputs on Ice Lake:

- [VPSHUFB_EVEX (YMM, YMM, YMM)](https://uops.info/html-instr/VPSHUFB_EVEX_YMM_YMM_YMM.html):
    - On Skylake-X: 1 cycle latency and executes only on port 5.
    - On Ice Lake: 1 cycle latency and executes on port 1 and 5.
    - On Zen4: 2 cycles and executes on both port 1 and 2.
- [VPDPBUSDS_EVEX (YMM, YMM, YMM)](https://uops.info/html-instr/VPDPBUSDS_EVEX_YMM_YMM_YMM.html):
    - On Ice Lake: 5 cycles latency and executes on both port 0 and 1.
    - On Zen4: 4 cycles and executes on both port 0 and 1.

So when implementing the Jaccard distance, the most important kernel for binary similarity indices using the `VPOPCNTQ`, we will have 4 such instruction invocations that will all stall on the same port number 5:

```c
float jaccard_distance_1024d(uint8_t const* a_vector, uint8_t const* b_vector) {
    __m512i a_start = _mm512_loadu_si512((__m512i const*)(a_vector));
    __m512i a_end = _mm512_loadu_si512((__m512i const*)(a_vector + 64));
    __m512i b_start = _mm512_loadu_si512((__m512i const*)(b_vector));
    __m512i b_end = _mm512_loadu_si512((__m512i const*)(b_vector + 64));
    __m512i intersection_start = _mm512_and_epi64(a_start, b_start);
    __m512i intersection_end = _mm512_and_epi64(a_end, b_end);
    __m512i union_start = _mm512_or_epi64(a_start, b_start);
    __m512i union_end = _mm512_or_epi64(a_end, b_end);
    __m512i population_intersection_start = _mm512_popcnt_epi64(intersection_start);
    __m512i population_intersection_end = _mm512_popcnt_epi64(intersection_end);
    __m512i population_union_start = _mm512_popcnt_epi64(union_start);
    __m512i population_union_end = _mm512_popcnt_epi64(union_end);
    __m512i population_intersection = _mm512_add_epi64(population_intersection_start, population_intersection_end);
    __m512i population_union = _mm512_add_epi64(population_union_start, population_union_end);
    return 1.f - _mm512_reduce_add_epi64(population_intersection) * 1.f / _mm512_reduce_add_epi64(population_union);
}
```

That's known to be a bottleneck and can be improved.

### Lookup Tables

A common trick is to implement population counts using lookup tables.
The idea may seem costly compared to `_mm512_popcnt_epi64`, but depends on the number of counts that need to be performed.
Here's what a minimal kernel for 245-dimensional Jaccard distance may look like:

```c
float jaccard_distance_256d(uint8_t const *a_vector, uint8_t const *b_vector) {
    __m256i a = _mm256_loadu_epi8((__m256i const*)(a_vector));
    __m256i b = _mm256_loadu_epi8((__m256i const*)(b_vector));
    
    __m256i intersection = _mm256_and_epi64(a, b);
    __m256i union_ = _mm256_or_epi64(a, b);

    __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i lookup = _mm256_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
    
    __m256i intersection_low = _mm256_and_si256(intersection, low_mask);
    __m256i intersection_high = _mm256_and_si256(_mm256_srli_epi16(intersection, 4), low_mask);
    __m256i union_low = _mm256_and_si256(union_, low_mask);
    __m256i union_high = _mm256_and_si256(_mm256_srli_epi16(union_, 4), low_mask);

    __m256i intersection_popcount = _mm256_add_epi8(
        _mm256_shuffle_epi8(lookup, intersection_low),
        _mm256_shuffle_epi8(lookup, intersection_high));
    __m256i union_popcount = _mm256_add_epi8(
        _mm256_shuffle_epi8(lookup, union_low),
        _mm256_shuffle_epi8(lookup, union_high));
    
    __m256i intersection_counts = _mm256_sad_epu8(intersection_popcount, _mm256_setzero_si256());
    __m256i union_counts = _mm256_sad_epu8(union_popcount, _mm256_setzero_si256());
    return 1.f - _mm256_reduce_add_epi64(intersection_counts) * 1.f / _mm256_reduce_add_epi64(union_counts);
}
```

The `_mm256_reduce_add_epi64` is a product of our imagination, but you can guess what it does: 

```c
uint64_t _mm256_reduce_add_epi64(__m256i x) {
    __m128i lo128 = _mm256_castsi256_si128(x);
    __m128i hi128 = _mm256_extracti128_si256(x, 1);
    __m128i sum128 = _mm_add_epi64(lo128, hi128);
    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    __m128i total = _mm_add_epi64(sum128, hi64);
    return uint64_t(_mm_cvtsi128_si64(total));
}
```

### Harley-Seal and Odd-Majority [CSA](https://en.wikipedia.org/wiki/Carry-save_adder)

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
X &:= \{x_0,x_1,x_2\} \\
\text{PopCount}(X) &:= \sum_{i=0}^{2}\text{PopCount}(x_i) \\
\text{Odd} &:= (x_0 \oplus x_1) \oplus\ x_2 \\
\text{Major} &:= \bigl((x_0 \oplus x_1)\land x_2\bigr) \lor\ (x_0 \land x_1) \\
\curvearrowright \text{PopCount}(X) &= 2 \text{PopCount}(\text{Major}) + \text{PopCount}(\text{Odd})
\end{aligned}
$$

That rule is an example of Carry-Save Adders (CSAs) and can be used for longer sequences as well.
So $N$ "Odd-Major" population counts can cover the logic needed for the $2^N-1$ original counts:

- 7x `std::popcount` can be reduced to 3x.
- 15x `std::popcount` can be reduced to 4x.
- 31x `std::popcount` can be reduced to 5x.

When dealing with 1024-dimensional bit-vectors on 64-bit machines, we can view them as 16x 64-bit words, leveraging the "Odd-Majority" trick to fold a 15x `std::popcount` into 4x, and then having 1x more call for the tail, thus shrinking from 16x to 5x calls, at the cost of several `XOR` and `AND` operations.
Sadly, even on Intel Sapphire Rapids, none of the CSA schemes result in gains compared to `VPOPCNTQ`:

```sh
Profiling `jaccard_b1536_vpopcntq` in USearch over 1,000 vectors
- BOP/S: 326.08 G
- Recall@1: 100.00%
Profiling `jaccard_b1536_vpopcntq_3csa` in USearch over 1,000 vectors
- BOP/S: 310.28 G
- Recall@1: 100.00%
```

## Links

- [Population Counts in Chess Engines](https://www.chessprogramming.org/Population_Count).
- [Faster Population Counts Using AVX2 Instructions](https://arxiv.org/abs/1611.07612), + [code](https://github.com/CountOnes/hamming_weight).
- [Carry-Save Adders](https://en.wikipedia.org/wiki/Carry-save_adder).
- [Tricks With the Floating-Point Format](https://randomascii.wordpress.com/2012/01/11/tricks-with-the-floating-point-format/)

#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "numpy",
#   "numba",
#   "cppyy",
#   "usearch",
#   "pytest",
#   "pytest-benchmark",
# ]
# ///

# ? Uses USearch native functionality for exact search with custom metrics
# ? to benchmark the efficiency of various Jaccard similarity implementations.
# ? Usage examples:
# ?
# ?   uv run --script kernels.py
# ?   uv run --script kernels.py --count 100 --ndims 1024
# ?   uv run --script kernels.py --count 10000 --ndims "512,1024,1536" --k 1
# ?
# ? The last example will compute 10K by 10K meaning 100M distances for 512-bit,
# ? 1024-bit and 1536-bit vectors. For each, only the top-1 nearest neighbor will
# ? be fetched.
from typing import List
import time

import cppyy
import cppyy.ll
import numpy as np
from numba import cfunc, types, carray
from numpy.ctypeslib import ndpointer
from usearch.index import (
    Index,
    CompiledMetric,
    MetricKind,
    MetricSignature,
    ScalarKind,
    BatchMatches,
    search,
)


def popcount_reduce_numpy(a: np.ndarray) -> np.ndarray:
    return np.unpackbits(a).astype(np.uint16).sum()


def jaccard_numpy(a: np.ndarray, b: np.ndarray) -> float:
    intersection = popcount_reduce_numpy(np.bitwise_and(a, b))
    union = popcount_reduce_numpy(np.bitwise_or(a, b))
    return 1.0 - (intersection + 1.0) / (union + 1.0)  # ! Avoid division by zero


@cfunc(types.uint64(types.uint64))
def popcount_u64_numba(v):
    v -= (v >> 1) & 0x5555555555555555
    v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0F
    v = (v * 0x0101010101010101) >> 56
    return v


@cfunc(types.float32(types.CPointer(types.uint8), types.CPointer(types.uint8)))
def jaccard_u8x128_numba(a, b):
    a_array = carray(a, 128)
    b_array = carray(b, 128)
    intersection = 0
    union = 0
    for i in range(128):
        intersection += popcount_u64_numba(a_array[i] & b_array[i])
        union += popcount_u64_numba(a_array[i] | b_array[i])
    return 1.0 - (intersection + 1.0) / (union + 1.0)  # ! Avoid division by zero


@cfunc(types.float32(types.CPointer(types.uint64), types.CPointer(types.uint64)))
def jaccard_u64x16_numba(a, b):
    a_array = carray(a, 16)
    b_array = carray(b, 16)
    intersection = 0
    union = 0
    for i in range(16):
        intersection += popcount_u64_numba(a_array[i] & b_array[i])
        union += popcount_u64_numba(a_array[i] | b_array[i])
    return 1.0 - (intersection + 1.0) / (union + 1.0)  # ! Avoid division by zero


jaccard_u8x128_cpp = """
static float jaccard_u8x128_cpp(uint8_t const * a, uint8_t const * b) {
    uint32_t intersection = 0, union_ = 0;
#pragma unroll
    for (size_t i = 0; i != 128; ++i)
        intersection += __builtin_popcount(a[i] & b[i]),
        union_ += __builtin_popcount(a[i] | b[i]);
    return 1.f - (intersection + 1.f) / (union_ + 1.f); // ! Avoid division by zero
}
"""

jaccard_u64x16_cpp = """
static float jaccard_u64x16_cpp(uint8_t const * a, uint8_t const * b) {
    uint32_t intersection = 0, union_ = 0;
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;
#pragma unroll
    for (size_t i = 0; i != 16; ++i)
        intersection += __builtin_popcountll(a64[i] & b64[i]),
        union_ += __builtin_popcountll(a64[i] | b64[i]);
    return 1.f - (intersection + 1.f) / (union_ + 1.f); // ! Avoid division by zero
}
"""

# Define the AVX-512 variant using the `vpopcntq` instruction.
# It's known to over-rely on port 5 on x86 CPUs, so the next `vpshufb` variant should be faster.
jaccard_b1024_vpopcntq = """
#include <immintrin.h>
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
static float jaccard_b1024_vpopcntq(uint8_t const * first_vector, uint8_t const * second_vector) {
    __m512i const first_start = _mm512_loadu_si512((__m512i const *)(first_vector));
    __m512i const first_end = _mm512_loadu_si512((__m512i const *)(first_vector + 64));
    __m512i const second_start = _mm512_loadu_si512((__m512i const *)(second_vector));
    __m512i const second_end = _mm512_loadu_si512((__m512i const *)(second_vector + 64));
    
    __m512i const intersection_start = _mm512_popcnt_epi64(_mm512_and_epi64(first_start, second_start));
    __m512i const intersection_end = _mm512_popcnt_epi64(_mm512_and_epi64(first_end, second_end));
    __m512i const union_start = _mm512_popcnt_epi64(_mm512_or_epi64(first_start, second_start));
    __m512i const union_end = _mm512_popcnt_epi64(_mm512_or_epi64(first_end, second_end));
    
    __m512i const intersection = _mm512_add_epi64(intersection_start, intersection_end);
    __m512i const union_ = _mm512_add_epi64(union_start, union_end);
    return 1.f - (_mm512_reduce_add_epi64(intersection) + 1.f) / (_mm512_reduce_add_epi64(union_) + 1.f);
}
"""

# Define the AVX-512 variant using the `vpshufb` instruction.
# It resorts to cheaper byte-shuffling instructions, than population counts.
# Source: https://github.com/CountOnes/hamming_weight/blob/1dd7554c0fc39e01c9d7fa54372fd4eccf458875/src/sse_jaccard_index.c#L17
jaccard_b1024_vpshufb = """
#include <immintrin.h>
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
static float jaccard_b1024_vpshufb(uint8_t const * first_vector, uint8_t const * second_vector) {
    __m512i const first_start = _mm512_loadu_si512((__m512i const *)(first_vector));
    __m512i const first_end = _mm512_loadu_si512((__m512i const *)(first_vector + 64));
    __m512i const second_start = _mm512_loadu_si512((__m512i const *)(second_vector));
    __m512i const second_end = _mm512_loadu_si512((__m512i const *)(second_vector + 64));
    
    __m512i const intersection_start = _mm512_and_epi64(first_start, second_start);
    __m512i const intersection_end = _mm512_and_epi64(first_end, second_end);
    __m512i const union_start = _mm512_or_epi64(first_start, second_start);
    __m512i const union_end = _mm512_or_epi64(first_end, second_end);

    __m512i const low_mask = _mm512_set1_epi8(0x0f);
    __m512i const lookup = _mm512_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
    
    __m512i const intersection_start_low = _mm512_and_si512(intersection_start, low_mask);
    __m512i const intersection_start_high = _mm512_and_si512(_mm512_srli_epi64(intersection_start, 4), low_mask);
    __m512i const intersection_end_low = _mm512_and_si512(intersection_end, low_mask);
    __m512i const intersection_end_high = _mm512_and_si512(_mm512_srli_epi64(intersection_end, 4), low_mask);

    __m512i const union_start_low = _mm512_and_si512(union_start, low_mask);
    __m512i const union_start_high = _mm512_and_si512(_mm512_srli_epi64(union_start, 4), low_mask);
    __m512i const union_end_low = _mm512_and_si512(union_end, low_mask);
    __m512i const union_end_high = _mm512_and_si512(_mm512_srli_epi64(union_end, 4), low_mask);

    __m512i const intersection_start_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, intersection_start_low),
        _mm512_shuffle_epi8(lookup, intersection_start_high));
    __m512i const intersection_end_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, intersection_end_low),
        _mm512_shuffle_epi8(lookup, intersection_end_high));
    __m512i const union_start_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, union_start_low),
        _mm512_shuffle_epi8(lookup, union_start_high));
    __m512i const union_end_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, union_end_low),
        _mm512_shuffle_epi8(lookup, union_end_high));
    
    __m512i const intersection = _mm512_add_epi64(
        _mm512_sad_epu8(intersection_start_popcount, _mm512_setzero_si512()), 
        _mm512_sad_epu8(intersection_end_popcount, _mm512_setzero_si512()));
    __m512i const union_ = _mm512_add_epi64(
        _mm512_sad_epu8(union_start_popcount, _mm512_setzero_si512()), 
        _mm512_sad_epu8(union_end_popcount, _mm512_setzero_si512()));
        
    return 1.f - (_mm512_reduce_add_epi64(intersection) + 1.f) / (_mm512_reduce_add_epi64(union_) + 1.f);
}
"""

cppyy.cppdef(jaccard_u8x128_cpp)
cppyy.cppdef(jaccard_u64x16_cpp)
cppyy.cppdef(jaccard_b1024_vpopcntq)
cppyy.cppdef(jaccard_b1024_vpshufb)


def generate_random_vectors(count: int, bits_per_vector: int) -> np.ndarray:
    bools = np.random.randint(0, 2, size=(count, bits_per_vector), dtype=np.uint8)
    bits = np.packbits(bools, axis=1)
    return bits


def bench_kernel(
    kernel_pointer: int,
    vectors: np.ndarray,
    k: int,
    threads: int,
    approximate: bool,
) -> dict:

    keys: np.ndarray = np.arange(vectors.shape[0], dtype=np.uint64)
    compiled_metric = CompiledMetric(
        pointer=kernel_pointer,
        kind=MetricKind.Tanimoto,
        signature=MetricSignature.ArrayArray,
    )

    start = time.perf_counter()
    bits_per_vector = vectors.shape[1] * 8

    matches = None
    if not approximate:
        matches: BatchMatches = search(
            metric=compiled_metric,
            dataset=vectors,
            query=vectors,
            count=k,  # ? Matches wanted per query
            exact=True,
            threads=threads,
        )
    else:
        index = Index(
            ndim=bits_per_vector,
            dtype=ScalarKind.B1,
            metric=compiled_metric,
        )
        index.add(keys, vectors, log=False)
        matches: BatchMatches = index.search(vectors, k, log=False)

    # Reduce stats
    elapsed_s = time.perf_counter() - start
    bit_ops_per_distance = bits_per_vector * 2
    recalled_top_match: int = matches.count_matches(keys, count=1)
    stats = {
        "visited_members": matches.visited_members,
        "computed_distances": matches.computed_distances,
        "elapsed_s": elapsed_s,
        "bit_ops_per_s": matches.computed_distances * bit_ops_per_distance / elapsed_s,
        "recalled_top_match": recalled_top_match,
    }

    return stats


def main(
    count: int,
    k: int = 1,
    ndims: List[int] = [1024],
    approximate: bool = True,
    threads: int = 1,
):

    kernels_cpp_1024d = [
        # C++:
        (
            "jaccard_u64x16_cpp",
            cppyy.gbl.jaccard_u64x16_cpp,
            cppyy.ll.addressof(cppyy.gbl.jaccard_u64x16_cpp),
        ),
        (
            "jaccard_u8x128_cpp",
            cppyy.gbl.jaccard_u8x128_cpp,
            cppyy.ll.addressof(cppyy.gbl.jaccard_u8x128_cpp),
        ),
        # SIMD:
        (
            "jaccard_b1024_vpopcntq",
            cppyy.gbl.jaccard_b1024_vpopcntq,
            cppyy.ll.addressof(cppyy.gbl.jaccard_b1024_vpopcntq),
        ),
        (
            "jaccard_b1024_vpshufb",
            cppyy.gbl.jaccard_b1024_vpshufb,
            cppyy.ll.addressof(cppyy.gbl.jaccard_b1024_vpshufb),
        ),
    ]
    kernels_numba_1024d = [
        # Baselines:
        (
            "jaccard_u64x16_numba",
            jaccard_u64x16_numba,
            jaccard_u64x16_numba.address,
        ),
        # ! Slower irrelevant kernels in the end if someone has the patience:
        (
            "jaccard_u8x128_numba",
            jaccard_u8x128_numba,
            jaccard_u8x128_numba.address,
        ),
    ]
    if 1024 in ndims:
        vectors = generate_random_vectors(count, 1024)

        # Run a few tests on this data:
        tests_per_kernel = 10
        for name, accelerated_kernel, _ in kernels_cpp_1024d:
            for _ in range(tests_per_kernel):
                first_vector_index = np.random.randint(0, count)
                second_vector_index = np.random.randint(0, count)
                first_vector, second_vector = (
                    vectors[first_vector_index],
                    vectors[second_vector_index],
                )
                baseline_distance = jaccard_numpy(first_vector, second_vector)
                accelerated_distance = accelerated_kernel(first_vector, second_vector)
                assert (
                    abs(baseline_distance - accelerated_distance) < 1e-5
                ), f"Distance mismatch for {name} kernel: {baseline_distance} vs {accelerated_distance}"

        # Run the benchmarks:
        for name, _, kernel_pointer in kernels_cpp_1024d + kernels_numba_1024d:
            print("-" * 80)
            print(f"Profiling `{name}` kernel over {count:,} vectors")
            stats = bench_kernel(
                kernel_pointer=kernel_pointer,
                vectors=vectors,
                k=k,
                approximate=approximate,
                threads=threads,
            )
            print()
            print(f"BOP/S: {stats['bit_ops_per_s'] / 1e9:,.2f} G")
            print(f"Recall@1: {stats['recalled_top_match'] / count:.2%}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    args = ArgumentParser()
    args.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of vectors to generate for the benchmark",
    )
    args.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of nearest neighbors to search for",
    )
    args.add_argument(
        "--ndims",
        type=int,
        nargs="+",
        default=[1024],
        help="List of dimensions to test (e.g., 128, 256, 512, 1024)",
    )
    args.add_argument(
        "--approximate",
        action="store_true",
        help="Use approximate search instead of exact search",
    )
    args = args.parse_args()
    main(
        count=args.count,
        k=args.k,
        ndims=args.ndims,
        approximate=args.approximate,
    )

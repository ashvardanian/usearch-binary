import os
import concurrent.futures
from typing import Optional

import pandas as pd
import fire

from usearch.io import (
    load_matrix,
    save_matrix,
)
from usearch.eval import (
    random_vectors,
    self_recall,
    SearchStats,
)
from usearch.index import (
    Index,
    CompiledMetric,
    MetricKind,
    MetricSignature,
    ScalarKind,
    BatchMatches,
    search,
)

# import cppyy
# import cppyy.ll
import numpy as np
from numba import cfunc, types, carray


@cfunc(types.uint64(types.uint64))
def popcount_numba(v):
    v -= (v >> 1) & 0x5555555555555555
    v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0F
    v = (v * 0x0101010101010101) >> 56
    return v


@cfunc(types.float32(types.CPointer(types.uint8), types.CPointer(types.uint8)))
def hamming_numba8bit(a, b):
    a_array = carray(a, 128)
    b_array = carray(b, 128)
    c = 0.0
    for i in range(128):
        c += popcount_numba(a_array[i] ^ b_array[i])
    return c


@cfunc(types.float32(types.CPointer(types.uint64), types.CPointer(types.uint64)))
def hamming_numba64bit(a, b):
    a_array = carray(a, 16)
    b_array = carray(b, 16)
    c = 0.0
    for i in range(16):
        c += popcount_numba(a_array[i] ^ b_array[i])
    return c


hamming_serial8bit = """
static float hamming_serial8bit(uint8_t const * a, uint8_t const * b) {
    uint32_t result = 0;
#pragma unroll
    for (size_t i = 0; i != 128; ++i)
        result += __builtin_popcount(a[i] ^ b[i]);
    return (double)result;
}
"""

hamming_serial64bit = """
static float hamming_serial64bit(uint8_t const * a, uint8_t const * b) {
    uint32_t result = 0;
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;
#pragma unroll
    for (size_t i = 0; i != 16; ++i)
        result += __builtin_popcountll(a64[i] ^ b64[i]);
    return (double)result;
}
"""

hamming_avx512_1024d = """
#include <immintrin.h>
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
static float hamming_avx512_1024d(uint8_t const * first_vector, uint8_t const * second_vector) {
    __m512i const first_start = _mm512_loadu_si512((__m512i const *)(first_vector));
    __m512i const first_end = _mm512_loadu_si512((__m512i const *)(first_vector + 64));
    __m512i const second_start = _mm512_loadu_si512((__m512i const *)(second_vector));
    __m512i const second_end = _mm512_loadu_si512((__m512i const *)(second_vector + 64));
    __m512i const differences_start = _mm512_xor_epi64(first_start, second_start);
    __m512i const differences_end = _mm512_xor_epi64(first_end, second_end);
    __m512i const population_start = _mm512_popcnt_epi64(differences_start);
    __m512i const population_end = _mm512_popcnt_epi64(differences_end);
    __m512i const population = _mm512_add_epi64(population_start, population_end);
    return (double)_mm512_reduce_add_epi64(population);
}
"""

jaccard_avx512_1024d = """
#include <immintrin.h>
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
static float jaccard_avx512_1024d(uint8_t const * first_vector, uint8_t const * second_vector) {
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
    return 1 - (double)_mm512_reduce_add_epi64(intersection) / _mm512_reduce_add_epi64(union_);
}
"""

hamming_avx512_768d = """
#include <immintrin.h>
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
static float hamming_avx512_768d(uint8_t const * first_vector, uint8_t const * second_vector) {
    __mmask8 const mask_end = 0xF0;
    __m512i const first_start = _mm512_loadu_si512((__m512i const *)(first_vector));
    __m512i const first_end = _mm512_maskz_loadu_epi64(mask_end, (__m512i const *)(first_vector + 64));
    __m512i const second_start = _mm512_loadu_si512((__m512i const *)(second_vector));
    __m512i const second_end = _mm512_maskz_loadu_epi64(mask_end, (__m512i const *)(second_vector + 64));
    __m512i const differences_start = _mm512_xor_epi64(first_start, second_start);
    __m512i const differences_end = _mm512_xor_epi64(first_end, second_end);
    __m512i const population_start = _mm512_popcnt_epi64(differences_start);
    __m512i const population_end = _mm512_popcnt_epi64(differences_end);
    __m512i const population = _mm512_add_epi64(population_start, population_end);
    return (double)_mm512_reduce_add_epi64(population);
}
"""

# cppyy.cppdef(hamming_serial8bit)
# cppyy.cppdef(hamming_serial64bit)
# cppyy.cppdef(hamming_avx512_1024d)
# cppyy.cppdef(jaccard_avx512_1024d)
# cppyy.cppdef(hamming_avx512_768d)


# Let's test our kernels
def test_hamming_functions():
    # Test vectors
    a = np.random.randint(0, 256, size=128, dtype=np.uint8)
    b = np.copy(a)  # Identical vectors should have a distance of 0
    c = np.copy(a)
    c[0] ^= 0xFF  # Invert all bits in the first byte; distance should be 8

    # Test each function
    serial_u8_result = cppyy.gbl.hamming_serial8bit(a, b)
    serial_u64_result = cppyy.gbl.hamming_serial64bit(a, b)
    avx512_result = cppyy.gbl.hamming_avx512_1024d(a, b)

    assert serial_u8_result == 0, "hamming_serial8bit failed on identical vectors"
    assert serial_u64_result == 0, "hamming_serial64bit failed on identical vectors"
    assert avx512_result == 0, "hamming_avx512_1024d failed on identical vectors"

    # Now test with vectors that should have a distance of 8
    serial_u8_result = cppyy.gbl.hamming_serial8bit(a, c)
    serial_u64_result = cppyy.gbl.hamming_serial64bit(a, c)
    avx512_result = cppyy.gbl.hamming_avx512_1024d(a, c)

    assert serial_u8_result == 8, "hamming_serial8bit failed on vectors with distance 8"
    assert (
        serial_u64_result == 8
    ), "hamming_serial64bit failed on vectors with distance 8"
    assert avx512_result == 8, "hamming_avx512_1024d failed on vectors with distance 8"

    print("All tests passed!")


def bench_kernel(
    kernel: int,
    vectors: np.ndarray,
    k: int = 10,
    exact: bool = False,
) -> SearchStats:

    keys: np.ndarray = np.arange(vectors.shape[0], dtype=np.uint64)
    compiled_metric = CompiledMetric(
        pointer=kernel,
        kind=MetricKind.Hamming,
        signature=MetricSignature.ArrayArray,
    )

    matches = None
    if exact:
        matches: BatchMatches = search(vectors, vectors, k, compiled_metric, exact=True)
    else:
        index = Index(
            ndim=1024,
            dtype=ScalarKind.B1,
            metric=compiled_metric,
        )
        index.add(keys, vectors, log=True)
        matches: BatchMatches = index.search(vectors, k, log=True)

    # Reduce stats
    count_correct: int = matches.count_matches(keys, count=k)
    return SearchStats(
        index_size=vectors.shape[0],
        count_queries=vectors.shape[0],
        count_matches=count_correct,
        visited_members=matches.visited_members,
        computed_distances=matches.computed_distances,
    )


def bench_kernels(vectors_count, k: int = 10, exact: bool = False):
    kernels = [
        # ("hamming_numba8bit", hamming_numba8bit.address),
        # ("hamming_numba64bit", hamming_numba64bit.address),
        # ("hamming_serial8bit", cppyy.ll.addressof(cppyy.gbl.hamming_serial8bit)),
        # ("hamming_serial64bit", cppyy.ll.addressof(cppyy.gbl.hamming_serial64bit)),
        ("hamming_avx512_1024d", cppyy.ll.addressof(cppyy.gbl.hamming_avx512_1024d)),
        ("jaccard_avx512_1024d", cppyy.ll.addressof(cppyy.gbl.jaccard_avx512_1024d)),
        # ("hamming_avx512_768d", cppyy.ll.addressof(cppyy.gbl.hamming_avx512_768d)),
    ]

    vectors: np.ndarray = random_vectors(
        vectors_count,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        ndim=1024,
    )

    for name, kernel in kernels:
        print("-" * 80)
        print(
            f"Profiling `{name}` kernel for {'exact' if exact else 'approx.'} search over {vectors_count:,} vectors"
        )
        stats = bench_kernel(kernel=kernel, vectors=vectors, k=k, exact=exact)
        print()
        print("- Mean recall: ", stats.mean_recall)
        print("- Mean efficiency: ", stats.mean_efficiency)
        print("-" * 80)


def read_embeddings(parquet_file_path: str, column: str = "emb"):
    print(f"Reading {parquet_file_path}")
    df = pd.read_parquet(parquet_file_path)
    emb = np.vstack(df[column].values, dtype=np.float16).copy()
    del df
    return emb


def fast_vstack(matrices: list):
    total_rows = sum(matrix.shape[0] for matrix in matrices)
    num_columns = matrices[0].shape[1]
    result = np.zeros((total_rows, num_columns), dtype=np.float16)

    # Copy each matrix into the result array
    current_row = 0
    for matrix in matrices:
        nrows = matrix.shape[0]
        np.copyto(result[current_row : current_row + nrows, :], matrix)
        current_row += nrows
    return result


def read_vectors(dir: str, column: str, limit: Optional[int] = None):
    if limit is not None:
        limit = int(limit)
        print(f"Limiting to {limit:,} vectors")

    # Go through the `dir` directory, reading `.parquet` files one after another.
    # From every file extract the `column` column and stack them vertically into a matrix,
    # until we reach `limit` rows.
    files = os.listdir(dir)
    files = [f for f in files if f.endswith(".parquet")]
    files.sort()

    # Assuming pre-processing can be quite expensive, let's start parallel processes
    # that would export all parquet files into hbin files.
    vectors = []
    if limit:
        count = 0
        for file in files:
            if limit is not None and count >= limit:
                break
            file_matrix = read_embeddings(os.path.join(dir, file), column)
            vectors.append(file_matrix)
            count += file_matrix.shape[0]

    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
            futures = []
            for file in files:
                futures.append(
                    executor.submit(read_embeddings, os.path.join(dir, file), column)
                )

            for future in concurrent.futures.as_completed(futures):
                file_matrix = future.result()
                vectors.append(file_matrix)

    # Let's stack now
    vectors = fast_vstack(vectors)
    if limit is not None:
        vectors = vectors[:limit]

    return vectors


def bench(
    dir: str = None,
    column: str = "emb",
    limit: Optional[int] = None,
    k: int = 10,
    exact: bool = False,
):

    # If no real data is provided, let's profile on synthetic data
    if dir is None:
        test_hamming_functions()
        bench_kernels(10_000, k=k, exact=True)
        bench_kernels(1_000_000, k=k, exact=False)
        return

    # Check if a non-empty cached matrix file exists
    vectors_filename_hbin = f"vectors-{limit}.hbin" if limit else "vectors.hbin"
    vectors = (
        read_vectors(dir, column, limit)
        if not os.path.exists(vectors_filename_hbin)
        or os.path.getsize(vectors_filename_hbin) == 0
        else load_matrix(vectors_filename_hbin)
    )
    print("Read vectors!")
    limit = vectors.shape[0]
    keys = np.arange(limit, dtype=np.uint64)
    print("Generated keys!")

    # Let's save the matrix locally
    save_matrix(vectors, vectors_filename_hbin)
    print("Saved vectors!")

    # Quantize into bits
    # vectors = vectors.astype(np.float16)  # don't need full resolution
    bit_vectors = np.packbits((vectors >= 0).astype(np.uint8), axis=1)
    print("Packed bit-vectors!")

    # The `self_recall` function contains a lot of serial code, that will take forever
    # for the entire collection. Let's limit the number of queries to 10M.
    limit_queries = 10_000_000
    keys_queries = (
        keys
        if len(keys) < limit_queries
        else np.random.choice(keys, limit_queries, replace=False)
    )
    print("Sampled queries!")

    # Now let's build a half-precision index and search for the top-10 nearest neighbors,
    # defining the baseline for the subsequent quantized experiments.
    print("-" * 80)
    print(f"Building `f16` index for {limit:,} vectors with Cosine metric")
    index = Index(ndim=1024, dtype="f16", metric="cos")
    index.add(keys, vectors, log=True)
    stats: SearchStats = self_recall(
        index,
        keys=keys_queries,
        vectors=vectors[keys_queries],
        count=k,
        exact=exact,
        log=True,
    )
    print()
    print("- Mean recall: ", stats.mean_recall)
    print("- Mean efficiency: ", stats.mean_efficiency)
    print(index.__repr_pretty__())
    print("-" * 80)

    # Now let's quantize into bits and search for the top-10 nearest neighbors.
    print("-" * 80)
    print(f"Building `b1` index for {limit:,} vectors with Hamming metric")
    index = Index(ndim=1024, dtype="b1", metric="hamming")
    index.add(keys, bit_vectors, log=True)
    stats: SearchStats = self_recall(
        index,
        keys=keys_queries,
        vectors=bit_vectors[keys_queries],
        count=k,
        exact=exact,
        log=True,
    )
    print()
    print("- Mean recall: ", stats.mean_recall)
    print("- Mean efficiency: ", stats.mean_efficiency)
    print(index.__repr_pretty__())
    print("-" * 80)

    # Let's check if Jaccard distance works better
    print("-" * 80)
    print(f"Building `b1` index for {limit:,} vectors with Jaccard metric")
    index = Index(ndim=1024, dtype="b1", metric="tanimoto")
    index.add(keys, bit_vectors, log=True)
    stats: SearchStats = self_recall(
        index,
        keys=keys_queries,
        vectors=bit_vectors[keys_queries],
        count=k,
        exact=exact,
        log=True,
    )
    print()
    print("- Mean recall: ", stats.mean_recall)
    print("- Mean efficiency: ", stats.mean_efficiency)
    print(index.__repr_pretty__())
    print("-" * 80)

    # Let's compare our kernels to naive slicing
    for ndim_slices in [16, 32, 64, 128, 256, 512]:
        print("-" * 80)
        print(
            f"Building `f16` index for {limit:,}x {ndim_slices}d vectors with Cosine metric"
        )
        index = Index(ndim=ndim_slices, dtype="f16", metric="cos")
        index.add(keys, vectors[:, :ndim_slices], log=True)
        stats: SearchStats = self_recall(
            index,
            keys=keys_queries,
            vectors=vectors[keys_queries, :ndim_slices],
            count=k,
            exact=exact,
            log=True,
        )
        print()
        print("- Mean recall: ", stats.mean_recall)
        print("- Mean efficiency: ", stats.mean_efficiency)
        print(index.__repr_pretty__())
        print("-" * 80)

    # Let's combine quantization with slicing
    for ndim_slices in [64, 128, 256, 512]:
        print("-" * 80)
        print(
            f"Building `b1` index for {limit:,}x {ndim_slices}d vectors with Hamming metric"
        )
        index = Index(ndim=ndim_slices, dtype="b1", metric="hamming")
        index.add(keys, bit_vectors[:, : ndim_slices // 8], log=True)  # 8 bits per byte
        stats: SearchStats = self_recall(
            index,
            keys=keys_queries,
            vectors=bit_vectors[keys_queries, : ndim_slices // 8],
            count=k,
            exact=exact,
            log=True,
        )
        print()
        print("- Mean recall: ", stats.mean_recall)
        print("- Mean efficiency: ", stats.mean_efficiency)
        print(index.__repr_pretty__())
        print("-" * 80)


if __name__ == "__main__":
    fire.Fire(bench)

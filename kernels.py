from usearch.eval import (
    random_vectors,
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

import cppyy
import cppyy.ll
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

cppyy.cppdef(hamming_serial8bit)
cppyy.cppdef(hamming_serial64bit)
cppyy.cppdef(hamming_avx512_1024d)
cppyy.cppdef(jaccard_avx512_1024d)
cppyy.cppdef(hamming_avx512_768d)


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


if __name__ == "__main__":
    test_hamming_functions()
    bench_kernels(10_000, exact=True)
    bench_kernels(1_000_000, exact=False)

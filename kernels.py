import cppyy
import cppyy.ll
import numpy as np

from usearch.eval import self_recall, random_vectors, SearchStats
from usearch.index import Index, CompiledMetric, MetricKind, MetricSignature, ScalarKind

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


# Let's test our kernels
def test_numba():
    # Test vectors
    a = np.random.randint(0, 256, size=128, dtype=np.uint8)
    b = np.copy(a)  # Identical vectors should have a distance of 0
    c = np.copy(a)
    c[0] ^= 0xFF  # Invert all bits in the first byte; distance should be 8

    # Test each function
    numba_result = hamming_numba8bit(a, b)
    assert numba_result == 0, "hamming_serial8bit failed on identical vectors"

    # Now test with vectors that should have a distance of 8
    numba_result = hamming_numba8bit(a, c)
    assert numba_result == 8, "hamming_serial8bit failed on vectors with distance 8"
    print("Numba tests passed!")


# test_numba()


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

hamming_avx512vpopcnt = """
#include <immintrin.h>
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512vpopcntdq")))
static float hamming_avx512vpopcnt(uint8_t const * first_vector, uint8_t const * second_vector) {
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

cppyy.cppdef(hamming_serial8bit)
cppyy.cppdef(hamming_serial64bit)
cppyy.cppdef(hamming_avx512vpopcnt)


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
    avx512_result = cppyy.gbl.hamming_avx512vpopcnt(a, b)

    assert serial_u8_result == 0, "hamming_serial8bit failed on identical vectors"
    assert serial_u64_result == 0, "hamming_serial64bit failed on identical vectors"
    assert avx512_result == 0, "hamming_avx512vpopcnt failed on identical vectors"

    # Now test with vectors that should have a distance of 8
    serial_u8_result = cppyy.gbl.hamming_serial8bit(a, c)
    serial_u64_result = cppyy.gbl.hamming_serial64bit(a, c)
    avx512_result = cppyy.gbl.hamming_avx512vpopcnt(a, c)

    assert serial_u8_result == 8, "hamming_serial8bit failed on vectors with distance 8"
    assert (
        serial_u64_result == 8
    ), "hamming_serial64bit failed on vectors with distance 8"
    assert avx512_result == 8, "hamming_avx512vpopcnt failed on vectors with distance 8"

    print("All tests passed!")


test_hamming_functions()

kernels = [
    # ("hamming_numba8bit", hamming_numba8bit.address),
    ("hamming_numba64bit", hamming_numba64bit.address),
    ("hamming_serial8bit", cppyy.ll.addressof(cppyy.gbl.hamming_serial8bit)),
    ("hamming_serial64bit", cppyy.ll.addressof(cppyy.gbl.hamming_serial64bit)),
    ("hamming_avx512vpopcnt", cppyy.ll.addressof(cppyy.gbl.hamming_avx512vpopcnt)),
]

vectors: np.ndarray = random_vectors(
    int(1e6),
    metric=MetricKind.Hamming,
    dtype=ScalarKind.B1,
    ndim=1024,
)

for name, kernel in kernels:
    print("-" * 80)
    print("Profiling kernel: ", name)
    compiled_metric = CompiledMetric(
        pointer=kernel,
        kind=MetricKind.Hamming,
        signature=MetricSignature.ArrayArray,
    )
    index = Index(
        ndim=1024,
        dtype=ScalarKind.B1,
        metric=compiled_metric,
    )

    keys: np.ndarray = np.arange(len(vectors), dtype=np.uint64)
    index.add(keys, vectors, log=True)

    stats: SearchStats = self_recall(index, exact=False, log=True)
    assert stats.visited_members > 0
    print()
    print("- Mean recall: ", stats.mean_recall)
    print("- Mean efficiency: ", stats.mean_efficiency)
    print("-" * 80)

    # stats: SearchStats = self_recall(index, exact=True, log=True)
    # assert stats.visited_members == 0, "Exact search won't attend index nodes"
    # assert stats.computed_distances == len(index), "And will compute the distance to every node"

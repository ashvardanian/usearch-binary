#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "numpy",
#   "numba",
#   "cppyy",
#   "usearch",
#   "faiss-cpu",
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
from typing import List, Literal
import time

import cppyy
import cppyy.ll
import numpy as np
from numba import cfunc, types, carray
from faiss import (
    METRIC_Jaccard as FAISS_METRIC_JACCARD,
    omp_set_num_threads as faiss_set_threads,
)
from faiss.contrib.exhaustive_search import knn as faiss_knn
from faiss.extra_wrappers import knn_hamming as faiss_knn_hamming
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


# region: 256d kernels


@cfunc(types.float32(types.CPointer(types.uint64), types.CPointer(types.uint64)))
def jaccard_u64x4_numba(a, b):
    a_array = carray(a, 4)
    b_array = carray(b, 4)
    intersection = 0
    union = 0
    for i in range(4):
        intersection += popcount_u64_numba(a_array[i] & b_array[i])
        union += popcount_u64_numba(a_array[i] | b_array[i])
    return 1.0 - (intersection + 1.0) / (union + 1.0)  # ! Avoid division by zero


jaccard_u64x4_c = """
static float jaccard_u64x4_c(uint8_t const * a, uint8_t const * b) {
    uint32_t intersection = 0, union_ = 0;
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;
#pragma unroll
    for (size_t i = 0; i != 4; ++i)
        intersection += __builtin_popcountll(a64[i] & b64[i]),
        union_ += __builtin_popcountll(a64[i] | b64[i]);
    return 1.f - (intersection + 1.f) / (union_ + 1.f); // ! Avoid division by zero
}
"""

# Define the AVX2 variant using the `vpshufb` and `vpsadbw` instruction.
# It resorts to cheaper byte-shuffling instructions, than population counts.
# Source: https://github.com/CountOnes/hamming_weight/blob/1dd7554c0fc39e01c9d7fa54372fd4eccf458875/src/sse_jaccard_index.c#L17
jaccard_b256_vpshufb_sad = """
#include <immintrin.h>

inline uint64_t _mm256_reduce_add_epi64(__m256i vec) {
    __m128i lo128 = _mm256_castsi256_si128(vec);
    __m128i hi128 = _mm256_extracti128_si256(vec, 1);
    __m128i sum128 = _mm_add_epi64(lo128, hi128);
    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    __m128i total = _mm_add_epi64(sum128, hi64);
    return uint64_t(_mm_cvtsi128_si64(total));
}

__attribute__((target("avx2,bmi2,avx")))
static float jaccard_b256_vpshufb_sad(uint8_t const * first_vector, uint8_t const * second_vector) {
    __m256i first = _mm256_loadu_epi8((__m256i const*)(first_vector));
    __m256i second = _mm256_loadu_epi8((__m256i const*)(second_vector));
    
    __m256i intersection = _mm256_and_epi64(first, second);
    __m256i union_ = _mm256_or_epi64(first, second);

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
    return 1.f - (_mm256_reduce_add_epi64(intersection_counts) + 1.f) / (_mm256_reduce_add_epi64(union_counts) + 1.f);
}
"""


# Define the AVX-512 variant using the `vpopcntq` instruction.
# It's known to over-rely on port 5 on x86 CPUs, so the next `vpshufb` variant should be faster.
jaccard_b256_vpopcntq = """
#include <immintrin.h>

__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
static float jaccard_b256_vpopcntq(uint8_t const * first_vector, uint8_t const * second_vector) {
    __m256i first = _mm256_loadu_epi8((__m256i const*)(first_vector));
    __m256i second = _mm256_loadu_epi8((__m256i const*)(second_vector));
    
    __m256i intersection = _mm256_popcnt_epi64(_mm256_and_epi64(first, second));
    __m256i union_ = _mm256_popcnt_epi64(_mm256_or_epi64(first, second));    
    return 1.f - (_mm256_reduce_add_epi64(intersection) + 1.f) / (_mm256_reduce_add_epi64(union_) + 1.f);
}
"""

cppyy.cppdef(jaccard_u64x4_c)
cppyy.cppdef(jaccard_b256_vpshufb_sad)
cppyy.cppdef(jaccard_b256_vpopcntq)

# endregion

# region: 1024d kernels


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


jaccard_u8x128_c = """
static float jaccard_u8x128_c(uint8_t const * a, uint8_t const * b) {
    uint32_t intersection = 0, union_ = 0;
#pragma unroll
    for (size_t i = 0; i != 128; ++i)
        intersection += __builtin_popcount(a[i] & b[i]),
        union_ += __builtin_popcount(a[i] | b[i]);
    return 1.f - (intersection + 1.f) / (union_ + 1.f); // ! Avoid division by zero
}
"""

jaccard_u64x16_c = """
static float jaccard_u64x16_c(uint8_t const * a, uint8_t const * b) {
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
    __m512i first_start = _mm512_loadu_si512((__m512i const*)(first_vector));
    __m512i first_end = _mm512_loadu_si512((__m512i const*)(first_vector + 64));
    __m512i second_start = _mm512_loadu_si512((__m512i const*)(second_vector));
    __m512i second_end = _mm512_loadu_si512((__m512i const*)(second_vector + 64));
    
    __m512i intersection_start = _mm512_popcnt_epi64(_mm512_and_epi64(first_start, second_start));
    __m512i intersection_end = _mm512_popcnt_epi64(_mm512_and_epi64(first_end, second_end));
    __m512i union_start = _mm512_popcnt_epi64(_mm512_or_epi64(first_start, second_start));
    __m512i union_end = _mm512_popcnt_epi64(_mm512_or_epi64(first_end, second_end));
    
    __m512i intersection = _mm512_add_epi64(intersection_start, intersection_end);
    __m512i union_ = _mm512_add_epi64(union_start, union_end);
    return 1.f - (_mm512_reduce_add_epi64(intersection) + 1.f) / (_mm512_reduce_add_epi64(union_) + 1.f);
}
"""

# Define the AVX-512 variant using the `vpshufb` and `vpsadbw` instruction.
# It resorts to cheaper byte-shuffling instructions, than population counts.
# Source: https://github.com/CountOnes/hamming_weight/blob/1dd7554c0fc39e01c9d7fa54372fd4eccf458875/src/sse_jaccard_index.c#L17
jaccard_b1024_vpshufb_sad = """
#include <immintrin.h>
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
static float jaccard_b1024_vpshufb_sad(uint8_t const * first_vector, uint8_t const * second_vector) {
    __m512i first_start = _mm512_loadu_si512((__m512i const*)(first_vector));
    __m512i first_end = _mm512_loadu_si512((__m512i const*)(first_vector + 64));
    __m512i second_start = _mm512_loadu_si512((__m512i const*)(second_vector));
    __m512i second_end = _mm512_loadu_si512((__m512i const*)(second_vector + 64));
    
    __m512i intersection_start = _mm512_and_epi64(first_start, second_start);
    __m512i intersection_end = _mm512_and_epi64(first_end, second_end);
    __m512i union_start = _mm512_or_epi64(first_start, second_start);
    __m512i union_end = _mm512_or_epi64(first_end, second_end);

    __m512i low_mask = _mm512_set1_epi8(0x0f);
    __m512i lookup = _mm512_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
    
    __m512i intersection_start_low = _mm512_and_si512(intersection_start, low_mask);
    __m512i intersection_start_high = _mm512_and_si512(_mm512_srli_epi16(intersection_start, 4), low_mask);
    __m512i intersection_end_low = _mm512_and_si512(intersection_end, low_mask);
    __m512i intersection_end_high = _mm512_and_si512(_mm512_srli_epi16(intersection_end, 4), low_mask);

    __m512i union_start_low = _mm512_and_si512(union_start, low_mask);
    __m512i union_start_high = _mm512_and_si512(_mm512_srli_epi16(union_start, 4), low_mask);
    __m512i union_end_low = _mm512_and_si512(union_end, low_mask);
    __m512i union_end_high = _mm512_and_si512(_mm512_srli_epi16(union_end, 4), low_mask);

    __m512i intersection_start_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, intersection_start_low),
        _mm512_shuffle_epi8(lookup, intersection_start_high));
    __m512i intersection_end_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, intersection_end_low),
        _mm512_shuffle_epi8(lookup, intersection_end_high));
    __m512i union_start_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, union_start_low),
        _mm512_shuffle_epi8(lookup, union_start_high));
    __m512i union_end_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, union_end_low),
        _mm512_shuffle_epi8(lookup, union_end_high));
    
    __m512i intersection = _mm512_add_epi64(
        _mm512_sad_epu8(intersection_start_popcount, _mm512_setzero_si512()), 
        _mm512_sad_epu8(intersection_end_popcount, _mm512_setzero_si512()));
    __m512i union_ = _mm512_add_epi64(
        _mm512_sad_epu8(union_start_popcount, _mm512_setzero_si512()), 
        _mm512_sad_epu8(union_end_popcount, _mm512_setzero_si512()));
        
    return 1.f - (_mm512_reduce_add_epi64(intersection) + 1.f) / (_mm512_reduce_add_epi64(union_) + 1.f);
}
"""

# Define the AVX-512 variant using the `vpshufb` and `vpdpbusd` instruction.
# It replaces the horizontal addition with a dot-product.
jaccard_b1024_vpshufb_dpb = """
#include <immintrin.h>
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
static float jaccard_b1024_vpshufb_dpb(uint8_t const * first_vector, uint8_t const * second_vector) {
    __m512i first_start = _mm512_loadu_si512((__m512i const*)(first_vector));
    __m512i first_end = _mm512_loadu_si512((__m512i const*)(first_vector + 64));
    __m512i second_start = _mm512_loadu_si512((__m512i const*)(second_vector));
    __m512i second_end = _mm512_loadu_si512((__m512i const*)(second_vector + 64));
    
    __m512i intersection_start = _mm512_and_epi64(first_start, second_start);
    __m512i intersection_end = _mm512_and_epi64(first_end, second_end);
    __m512i union_start = _mm512_or_epi64(first_start, second_start);
    __m512i union_end = _mm512_or_epi64(first_end, second_end);

    __m512i ones = _mm512_set1_epi8(1);
    __m512i low_mask = _mm512_set1_epi8(0x0f);
    __m512i lookup = _mm512_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
    
    __m512i intersection_start_low = _mm512_and_si512(intersection_start, low_mask);
    __m512i intersection_start_high = _mm512_and_si512(_mm512_srli_epi16(intersection_start, 4), low_mask);
    __m512i intersection_end_low = _mm512_and_si512(intersection_end, low_mask);
    __m512i intersection_end_high = _mm512_and_si512(_mm512_srli_epi16(intersection_end, 4), low_mask);

    __m512i union_start_low = _mm512_and_si512(union_start, low_mask);
    __m512i union_start_high = _mm512_and_si512(_mm512_srli_epi16(union_start, 4), low_mask);
    __m512i union_end_low = _mm512_and_si512(union_end, low_mask);
    __m512i union_end_high = _mm512_and_si512(_mm512_srli_epi16(union_end, 4), low_mask);

    __m512i intersection_start_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, intersection_start_low),
        _mm512_shuffle_epi8(lookup, intersection_start_high));
    __m512i intersection_end_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, intersection_end_low),
        _mm512_shuffle_epi8(lookup, intersection_end_high));
    __m512i union_start_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, union_start_low),
        _mm512_shuffle_epi8(lookup, union_start_high));
    __m512i union_end_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, union_end_low),
        _mm512_shuffle_epi8(lookup, union_end_high));
    
    __m512i intersection = _mm512_dpbusd_epi32(_mm512_setzero_si512(), intersection_start_popcount, ones);
    intersection = _mm512_dpbusd_epi32(intersection, intersection_end_popcount, ones);

    __m512i union_ = _mm512_dpbusd_epi32(_mm512_setzero_si512(), union_start_popcount, ones);
    union_ = _mm512_dpbusd_epi32(union_, union_end_popcount, ones);
            
    return 1.f - (_mm512_reduce_add_epi32(intersection) + 1.f) / (_mm512_reduce_add_epi32(union_) + 1.f);
}
"""

# Harley-Seal transformation and Odd-Major-style Carry-Save-Adders can be used to replace
# several population counts with a few bitwise operations and one `popcount`, which can help
# lift the pressure on the CPU ports.
jaccard_u64x16_csa3_c = """
inline int popcount_csa3(uint64_t x, uint64_t y, uint64_t z) {
    uint64_t odd  = (x ^ y) ^ z;
    uint64_t major = ((x ^ y) & z) | (x & y);
    return 2 * __builtin_popcountll(major) + __builtin_popcountll(odd);
}

static float jaccard_u64x16_csa3_c(uint8_t const * a, uint8_t const * b) {
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;
    
    int intersection =
        popcount_csa3(a64[0] & b64[0], a64[1] & b64[1], a64[2] & b64[2]) +
        popcount_csa3(a64[3] & b64[3], a64[4] & b64[4], a64[5] & b64[5]) +
        popcount_csa3(a64[6] & b64[6], a64[7] & b64[7], a64[8] & b64[8]) +
        popcount_csa3(a64[9] & b64[9], a64[10] & b64[10], a64[11] & b64[11]) +
        popcount_csa3(a64[12] & b64[12], a64[13] & b64[13], a64[14] & b64[14]) +
        __builtin_popcountll(a64[15] & b64[15]);
    
        
    int union_ =
        popcount_csa3(a64[0] | b64[0], a64[1] | b64[1], a64[2] | b64[2]) +
        popcount_csa3(a64[3] | b64[3], a64[4] | b64[4], a64[5] | b64[5]) +
        popcount_csa3(a64[6] | b64[6], a64[7] | b64[7], a64[8] | b64[8]) +
        popcount_csa3(a64[9] | b64[9], a64[10] | b64[10], a64[11] | b64[11]) +
        popcount_csa3(a64[12] | b64[12], a64[13] | b64[13], a64[14] | b64[14]) +
        __builtin_popcountll(a64[15] | b64[15]);
    
    return 1.f - (intersection + 1.f) / (union_ + 1.f); // ! Avoid division by zero
}
"""

# That CSA can be scaled further to fold 15 population counts into 4.
# It's a bit more complex and for readability we will use C++ tuple unpacking:
jaccard_u64x16_csa15_cpp = """
struct uint64_csa_t {
   uint64_t ones;
   uint64_t twos;
};

constexpr uint64_csa_t csa(uint64_t x, uint64_t y, uint64_t z) {
    uint64_t odd  = (x ^ y) ^ z;
    uint64_t major = ((x ^ y) & z) | (x & y);
    return {odd, major};
}

constexpr int popcount_csa15(
    uint64_t x1, uint64_t x2, uint64_t x3,
    uint64_t x4, uint64_t x5, uint64_t x6, uint64_t x7,
    uint64_t x8, uint64_t x9, uint64_t x10, uint64_t x11,
    uint64_t x12, uint64_t x13, uint64_t x14, uint64_t x15) {
        
    auto [one1, two1] = csa(x1,  x2,  x3);
    auto [one2, two2] = csa(x4,  x5,  x6);
    auto [one3, two3] = csa(x7,  x8,  x9);
    auto [one4, two4] = csa(x10, x11, x12);
    auto [one5, two5] = csa(x13, x14, x15);

    // Level‐2: fold the five “one” terms down to two + a final “ones”
    auto [one6, two6] = csa(one1, one2, one3);
    auto [ones, two7] = csa(one4, one5, one6);

    // Level‐2: fold the five “two” terms down to two + a “twos”
    auto [two8, four1] = csa(two1, two2, two3);
    auto [two9, four2] = csa(two4, two5, two6);
    auto [twos, four3] = csa(two7, two8, two9);

    // Level‐3: fold the three “four” terms down to one “four” + one “eight”
    auto [four, eight] = csa(four1, four2, four3);

    // Now you have a full 4-bit per-bit‐position counter in (ones, twos, four, eight).
    int count_ones  = __builtin_popcountll(ones);
    int count_twos  = __builtin_popcountll(twos);
    int count_four  = __builtin_popcountll(four);
    int count_eight = __builtin_popcountll(eight);
    return count_ones + 2 * count_twos + 4 * count_four + 8 * count_eight;
}

static float jaccard_u64x16_csa15_cpp(uint8_t const * a, uint8_t const * b) {
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;
    
    int intersection = popcount_csa15(
        a64[0] & b64[0], a64[1] & b64[1], a64[2] & b64[2], a64[3] & b64[3],
        a64[4] & b64[4], a64[5] & b64[5], a64[6] & b64[6], a64[7] & b64[7],
        a64[8] & b64[8], a64[9] & b64[9], a64[10] & b64[10], a64[11] & b64[11],
        a64[12] & b64[12], a64[13] & b64[13], a64[14] & b64[14]) +
        __builtin_popcountll(a64[15] & b64[15]);
    
    int union_ = popcount_csa15(
        a64[0] | b64[0], a64[1] | b64[1], a64[2] | b64[2], a64[3] | b64[3],
        a64[4] | b64[4], a64[5] | b64[5], a64[6] | b64[6], a64[7] | b64[7],
        a64[8] | b64[8], a64[9] | b64[9], a64[10] | b64[10], a64[11] | b64[11],
        a64[12] | b64[12], a64[13] | b64[13], a64[14] | b64[14]) +
        __builtin_popcountll(a64[15] | b64[15]);
    
    return 1.f - (intersection + 1.f) / (union_ + 1.f); // ! Avoid division by zero
}
"""

cppyy.cppdef(jaccard_u8x128_c)
cppyy.cppdef(jaccard_u64x16_c)
cppyy.cppdef(jaccard_b1024_vpopcntq)
cppyy.cppdef(jaccard_b1024_vpshufb_sad)
cppyy.cppdef(jaccard_b1024_vpshufb_dpb)
cppyy.cppdef(jaccard_u64x16_csa3_c)
cppyy.cppdef(jaccard_u64x16_csa15_cpp)

# endregion

# region: 1536d kernels


@cfunc(types.float32(types.CPointer(types.uint64), types.CPointer(types.uint64)))
def jaccard_u64x24_numba(a, b):
    a_array = carray(a, 24)
    b_array = carray(b, 24)
    intersection = 0
    union = 0
    for i in range(24):
        intersection += popcount_u64_numba(a_array[i] & b_array[i])
        union += popcount_u64_numba(a_array[i] | b_array[i])
    return 1.0 - (intersection + 1.0) / (union + 1.0)  # ! Avoid division by zero


jaccard_u64x24_c = """
static float jaccard_u64x24_c(uint8_t const * a, uint8_t const * b) {
    uint32_t intersection = 0, union_ = 0;
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;
#pragma unroll
    for (size_t i = 0; i != 24; ++i)
        intersection += __builtin_popcountll(a64[i] & b64[i]),
        union_ += __builtin_popcountll(a64[i] | b64[i]);
    return 1.f - (intersection + 1.f) / (union_ + 1.f); // ! Avoid division by zero
}
"""

# Define the AVX-512 variant using the `vpopcntq` instruction for 1536d vectors
# It's known to over-rely on port 5 on x86 CPUs, so the next `vpshufb` variant should be faster.
jaccard_b1536_vpopcntq = """
#include <immintrin.h>
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
static float jaccard_b1536_vpopcntq(uint8_t const * first_vector, uint8_t const * second_vector) {
    __m512i first0 = _mm512_loadu_si512((__m512i const*)(first_vector + 64 * 0));
    __m512i first1 = _mm512_loadu_si512((__m512i const*)(first_vector + 64 * 1));
    __m512i first2 = _mm512_loadu_si512((__m512i const*)(first_vector + 64 * 2));
    __m512i second0 = _mm512_loadu_si512((__m512i const*)(second_vector + 64 * 0));
    __m512i second1 = _mm512_loadu_si512((__m512i const*)(second_vector + 64 * 1));
    __m512i second2 = _mm512_loadu_si512((__m512i const*)(second_vector + 64 * 2));
    
    __m512i intersection0 = _mm512_popcnt_epi64(_mm512_and_epi64(first0, second0));
    __m512i intersection1 = _mm512_popcnt_epi64(_mm512_and_epi64(first1, second1));
    __m512i intersection2 = _mm512_popcnt_epi64(_mm512_and_epi64(first2, second2));
    __m512i union0 = _mm512_popcnt_epi64(_mm512_or_epi64(first0, second0));
    __m512i union1 = _mm512_popcnt_epi64(_mm512_or_epi64(first1, second1));
    __m512i union2 = _mm512_popcnt_epi64(_mm512_or_epi64(first2, second2));
    
    __m512i intersection = _mm512_add_epi64(_mm512_add_epi64(intersection0, intersection1), intersection2);
    __m512i union_ = _mm512_add_epi64(_mm512_add_epi64(union0, union1), union2);
    return 1.f - (_mm512_reduce_add_epi64(intersection) + 1.f) / (_mm512_reduce_add_epi64(union_) + 1.f);
}
"""

# Define the AVX-512 variant, combining Harley-Seal transform to reduce the number
# of population counts for the 1536-dimensional case to the 1024-dimensional case,
# at the cost of several ternary bitwise operations.
jaccard_b1536_vpopcntq_3csa = """
#include <immintrin.h>
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
static float jaccard_b1536_vpopcntq_3csa(uint8_t const * first_vector, uint8_t const * second_vector) {

    __m512i first0 = _mm512_loadu_si512((__m512i const*)(first_vector + 64 * 0));
    __m512i first1 = _mm512_loadu_si512((__m512i const*)(first_vector + 64 * 1));
    __m512i first2 = _mm512_loadu_si512((__m512i const*)(first_vector + 64 * 2));
    __m512i second0 = _mm512_loadu_si512((__m512i const*)(second_vector + 64 * 0));
    __m512i second1 = _mm512_loadu_si512((__m512i const*)(second_vector + 64 * 1));
    __m512i second2 = _mm512_loadu_si512((__m512i const*)(second_vector + 64 * 2));
    
    __m512i intersection0 = _mm512_and_epi64(first0, second0);
    __m512i intersection1 = _mm512_and_epi64(first1, second1);
    __m512i intersection2 = _mm512_and_epi64(first2, second2);
    __m512i union0 = _mm512_or_epi64(first0, second0);
    __m512i union1 = _mm512_or_epi64(first1, second1);
    __m512i union2 = _mm512_or_epi64(first2, second2);
    
    __m512i intersection_odd = _mm512_ternarylogic_epi64(
        intersection0, intersection1, intersection2, 
        (_MM_TERNLOG_A ^ _MM_TERNLOG_B ^ _MM_TERNLOG_C));
    __m512i intersection_major = _mm512_ternarylogic_epi64(
        intersection0, intersection1, intersection2, 
        ((_MM_TERNLOG_A ^ _MM_TERNLOG_B) & _MM_TERNLOG_C) | (_MM_TERNLOG_A & _MM_TERNLOG_B));
    __m512i union_odd = _mm512_ternarylogic_epi64(
        union0, union1, union2, 
        (_MM_TERNLOG_A ^ _MM_TERNLOG_B ^ _MM_TERNLOG_C));
    __m512i union_major = _mm512_ternarylogic_epi64(
        union0, union1, union2, 
        ((_MM_TERNLOG_A ^ _MM_TERNLOG_B) & _MM_TERNLOG_C) | (_MM_TERNLOG_A & _MM_TERNLOG_B));
    
    __m512i intersection_odd_count = _mm512_popcnt_epi64(intersection_odd);
    __m512i intersection_major_count = _mm512_popcnt_epi64(intersection_major);
    __m512i union_odd_count = _mm512_popcnt_epi64(union_odd);
    __m512i union_major_count = _mm512_popcnt_epi64(union_major);

    // Shift left the majors by 1 to multiply by 2
    __m512i intersection = _mm512_add_epi64(_mm512_slli_epi64(intersection_major_count, 1), intersection_odd_count);
    __m512i union_ = _mm512_add_epi64(_mm512_slli_epi64(union_major_count, 1), union_odd_count);
    return 1.f - (_mm512_reduce_add_epi64(intersection) + 1.f) / (_mm512_reduce_add_epi64(union_) + 1.f);
}
"""

cppyy.cppdef(jaccard_u64x24_c)
cppyy.cppdef(jaccard_b1536_vpopcntq)
cppyy.cppdef(jaccard_b1536_vpopcntq_3csa)

# endregion


def generate_random_vectors(count: int, bits_per_vector: int) -> np.ndarray:
    bools = np.random.randint(0, 2, size=(count, bits_per_vector), dtype=np.uint8)
    bits = np.packbits(bools, axis=1)
    return bits


def bench_faiss(
    vectors: np.ndarray,
    k: int,
    threads: int,
    metric: Literal["Hamming", "Jaccard"],
) -> dict:

    faiss_set_threads(threads)
    n = vectors.shape[0]
    start = time.perf_counter()
    if metric == "Jaccard":
        _, matches = faiss_knn(vectors, vectors, k, metric=FAISS_METRIC_JACCARD)
    else:
        _, matches = faiss_knn_hamming(vectors, vectors, k)
    elapsed = time.perf_counter() - start

    computed = n * n
    recalled_top_match = int((matches[:, 0] == np.arange(n)).sum())
    bop_per_dist = vectors.shape[1] * (8 if vectors.dtype == np.uint8 else 1) * 2
    return {
        "elapsed_s": elapsed,
        "computed_distances": computed,
        "visited_members": computed,
        "bit_ops_per_s": computed * bop_per_dist / elapsed,
        "recalled_top_match": recalled_top_match,
    }


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
    return {
        "visited_members": matches.visited_members,
        "computed_distances": matches.computed_distances,
        "elapsed_s": elapsed_s,
        "bit_ops_per_s": matches.computed_distances * bit_ops_per_distance / elapsed_s,
        "recalled_top_match": recalled_top_match,
    }


def main(
    count: int,
    k: int = 1,
    ndims: List[int] = [256, 1024, 1536],
    approximate: bool = True,
    threads: int = 1,
):

    kernels_cpp_256d = [
        # C++:
        (
            "jaccard_u64x4_c",
            cppyy.gbl.jaccard_u64x4_c,
            cppyy.ll.addressof(cppyy.gbl.jaccard_u64x4_c),
        ),
        (
            "jaccard_b256_vpshufb_sad",
            cppyy.gbl.jaccard_b256_vpshufb_sad,
            cppyy.ll.addressof(cppyy.gbl.jaccard_b256_vpshufb_sad),
        ),
        (
            "jaccard_b256_vpopcntq",
            cppyy.gbl.jaccard_b256_vpopcntq,
            cppyy.ll.addressof(cppyy.gbl.jaccard_b256_vpopcntq),
        ),
    ]
    kernels_numba_256d = [
        # Baselines:
        (
            "jaccard_u64x4_numba",
            jaccard_u64x4_numba,
            jaccard_u64x4_numba.address,
        ),
    ]

    kernels_cpp_1024d = [
        # C++:
        (
            "jaccard_u64x16_c",
            cppyy.gbl.jaccard_u64x16_c,
            cppyy.ll.addressof(cppyy.gbl.jaccard_u64x16_c),
        ),
        (
            "jaccard_u8x128_c",
            cppyy.gbl.jaccard_u8x128_c,
            cppyy.ll.addressof(cppyy.gbl.jaccard_u8x128_c),
        ),
        (
            "jaccard_u64x16_csa3_c",
            cppyy.gbl.jaccard_u64x16_csa3_c,
            cppyy.ll.addressof(cppyy.gbl.jaccard_u64x16_csa3_c),
        ),
        (
            "jaccard_u64x16_csa15_cpp",
            cppyy.gbl.jaccard_u64x16_csa15_cpp,
            cppyy.ll.addressof(cppyy.gbl.jaccard_u64x16_csa15_cpp),
        ),
        # SIMD:
        (
            "jaccard_b1024_vpopcntq",
            cppyy.gbl.jaccard_b1024_vpopcntq,
            cppyy.ll.addressof(cppyy.gbl.jaccard_b1024_vpopcntq),
        ),
        (
            "jaccard_b1024_vpshufb_sad",
            cppyy.gbl.jaccard_b1024_vpshufb_sad,
            cppyy.ll.addressof(cppyy.gbl.jaccard_b1024_vpshufb_sad),
        ),
        (
            "jaccard_b1024_vpshufb_dpb",
            cppyy.gbl.jaccard_b1024_vpshufb_dpb,
            cppyy.ll.addressof(cppyy.gbl.jaccard_b1024_vpshufb_dpb),
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
    kernels_cpp_1536d = [
        # C++:
        (
            "jaccard_u64x24_c",
            cppyy.gbl.jaccard_u64x24_c,
            cppyy.ll.addressof(cppyy.gbl.jaccard_u64x24_c),
        ),
        # SIMD:
        (
            "jaccard_b1536_vpopcntq",
            cppyy.gbl.jaccard_b1536_vpopcntq,
            cppyy.ll.addressof(cppyy.gbl.jaccard_b1536_vpopcntq),
        ),
        (
            "jaccard_b1536_vpopcntq_3csa",
            cppyy.gbl.jaccard_b1536_vpopcntq_3csa,
            cppyy.ll.addressof(cppyy.gbl.jaccard_b1536_vpopcntq_3csa),
        ),
    ]
    kernels_numba_1536d = [
        # Baselines:
        (
            "jaccard_u64x24_numba",
            jaccard_u64x24_numba,
            jaccard_u64x24_numba.address,
        ),
    ]

    # Group kernels by dimension:
    kernels_cpp_per_dimension = {
        256: kernels_cpp_256d,
        1024: kernels_cpp_1024d,
        1536: kernels_cpp_1536d,
    }
    kernels_numba_per_dimension = {
        256: kernels_numba_256d,
        1024: kernels_numba_1024d,
        1536: kernels_numba_1536d,
    }

    # Check which dimensions should be covered:
    for ndim in ndims:
        print("-" * 80)
        print(f"Testing {ndim:,}d kernels")
        kernels_cpp = kernels_cpp_per_dimension.get(ndim, [])
        kernels_numba = kernels_numba_per_dimension.get(ndim, [])
        vectors = generate_random_vectors(count, ndim)

        # Run a few tests on this data:
        tests_per_kernel = 10
        for name, accelerated_kernel, _ in kernels_cpp:
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

        print("- passed!")

        # Provide FAISS benchmarking baselines:
        for name in ["Jaccard", "Hamming"]:
            print(f"Profiling FAISS over {count:,} vectors with {name} metric")
            stats = bench_faiss(
                vectors=vectors,
                k=k,
                threads=threads,
                metric=name,
            )
            print(f"- BOP/S: {stats['bit_ops_per_s'] / 1e9:,.2f} G")
            print(f"- Recall@1: {stats['recalled_top_match'] / count:.2%}")

        # Analyze all the kernels:
        for name, _, kernel_pointer in kernels_cpp + kernels_numba:
            print(f"Profiling `{name}` in USearch over {count:,} vectors")
            stats = bench_kernel(
                kernel_pointer=kernel_pointer,
                vectors=vectors,
                k=k,
                approximate=approximate,
                threads=threads,
            )
            print(f"- BOP/S: {stats['bit_ops_per_s'] / 1e9:,.2f} G")
            print(f"- Recall@1: {stats['recalled_top_match'] / count:.2%}")


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
        default=[256, 1024, 1536],
        help="List of dimensions to test (e.g., 256, 1024, 1536)",
    )
    args.add_argument(
        "--approximate",
        action="store_true",
        help="Use approximate search instead of exact search",
    )
    args.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use for the benchmark",
    )
    args = args.parse_args()
    main(
        count=args.count,
        k=args.k,
        ndims=args.ndims,
        approximate=args.approximate,
        threads=args.threads,
    )

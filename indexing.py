#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "numpy",
#   "usearch",
#   "pyarrow",
#   "pandas",
#   "tqdm",
# ]
# ///

# ? Uses USearch to incrementally construct bit-level and floating-point indexes
# ? for the same real-world data and evaluate both throughput and recall.
# ? Collections of Parquet files are used as inputs.
# ?
# ?   uv run --script indexing.py --metric "hamming"
# ?   uv run --script indexing.py --data "cohere/" --vector-column "emb" --k 10
# ?   uv run --script indexing.py --data "cohere/" --dtype "bf16" --ndim 64 --limit 1e6
# ?
# ? The last example will use BrainFloat16 (bf16) instead of larger single-precision
# ? floats for the baseline index. It will also only take the first 64 dimensions of
# ? input vector "Matryoshka"-style.
# ? All of the files will be unpacked on the fly, but that won't affect the timings.
import os
import concurrent
from os import PathLike
from typing import List, Literal, Optional

import pandas as pd
import numpy as np
from usearch.eval import (
    self_recall,
    SearchStats,
)
from usearch.index import (
    Index,
)


def fast_vstack(matrices: List[np.ndarray]) -> np.ndarray:
    """Faster alternative to `np.vstack` for large matrices."""
    total_rows = sum(matrix.shape[0] for matrix in matrices)
    num_columns = matrices[0].shape[1]
    result = np.zeros((total_rows, num_columns), dtype=matrices[0].dtype)

    # Copy each matrix into the result array
    current_row = 0
    for matrix in matrices:
        nrows = matrix.shape[0]
        np.copyto(result[current_row : current_row + nrows, :], matrix)
        current_row += nrows
    return result


def load_embeddings_column(
    parquet_path: PathLike,
    column: str = "emb",
    dtype=np.float32,
) -> np.ndarray:
    """Reads a column of vector from a Parquet file and returns as a NumPy matrix."""
    print(f"Reading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    emb = np.vstack(df[column].values, dtype=dtype).copy()
    del df
    return emb


def load_embeddings_in_parallel(
    dir: str,
    column: str,
    dtype=np.float32,
    limit: Optional[int] = None,
) -> np.ndarray:

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
    # that would export all Parquet files into `.hbin`` files.
    vectors = []
    count = 0
    if limit is not None:
        for file in files:
            if count >= limit:
                break
            file_matrix = load_embeddings_column(
                os.path.join(dir, file),
                column,
                dtype,
            )
            vectors.append(file_matrix)
            count += file_matrix.shape[0]

    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
            futures = []
            for file in files:
                futures.append(
                    executor.submit(
                        load_embeddings_column,
                        os.path.join(dir, file),
                        column,
                        dtype,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                file_matrix = future.result()
                vectors.append(file_matrix)

    # Let's stack now
    vectors = fast_vstack(vectors)
    if limit is not None:
        vectors = vectors[:limit]

    return vectors


def main(
    data: PathLike,
    dtype: Literal["float32", "float16", "int8"] = "float32",
    vector_column: str = "emb",
    metric: Literal["jaccard", "hamming"] = "jaccard",
    k: int = 1,
    ndim: Optional[int] = None,
    limit: Optional[int] = None,
):

    limit = int(limit) if limit is not None else None
    ndim = int(ndim) if ndim is not None else None
    assert dtype in ["float32", "float16", "int8"]
    assert metric in ["jaccard", "hamming"]
    assert k > 0

    # Load dataset
    embeddings = load_embeddings_in_parallel(
        data,
        column=vector_column,
        limit=limit,
        dtype=dtype,
    )
    if ndim is not None:
        embeddings = embeddings[:, :ndim]
    else:
        ndim = embeddings.shape[1]

    # Quantize into bits
    embeddings_binary = np.packbits((embeddings >= 0).astype(np.uint8), axis=1)
    keys = np.arange(limit, dtype=np.uint64)

    # The `self_recall` function contains a lot of serial code, that will take forever
    # for the entire collection. Let's limit the number of queries to 10M.
    limit_queries = 10_000_000
    keys_queries = (
        keys
        if len(keys) < limit_queries
        else np.random.choice(keys, limit_queries, replace=False)
    )

    # Now let's build a half-precision index and search for the top-10 nearest neighbors,
    # defining the baseline for the subsequent quantized experiments.
    print("-" * 80)
    print(f"Building `{dtype}` index for {limit:,} vectors with Cosine metric")
    index = Index(ndim=ndim, dtype=dtype, metric="cos")
    index.add(keys, embeddings, log=True)
    stats: SearchStats = self_recall(
        index,
        keys=keys_queries,
        vectors=embeddings[keys_queries],
        count=k,
        log=True,
    )
    print()
    print("- Mean recall: ", stats.mean_recall)
    print("- Mean efficiency: ", stats.mean_efficiency)
    print(index.__repr_pretty__())
    print("-" * 80)

    # Now let's quantize into bits and search for the top-10 nearest neighbors.
    print("-" * 80)
    metric_name = metric.capitalize()
    print(f"Building `b1` index for {limit:,} vectors with {metric_name} metric")
    index = Index(ndim=ndim, dtype="b1", metric=metric)
    index.add(keys, embeddings_binary, log=True)
    stats: SearchStats = self_recall(
        index,
        keys=keys_queries,
        vectors=embeddings_binary[keys_queries],
        count=k,
        log=True,
    )
    print()
    print("- Mean recall: ", stats.mean_recall)
    print("- Mean efficiency: ", stats.mean_efficiency)
    print(index.__repr_pretty__())
    print("-" * 80)


if __name__ == "__main__":
    from argparse import ArgumentParser

    arg_parser = ArgumentParser(description="Floats vs. Bits with USearch")
    arg_parser.add_argument(
        "--data",
        type=str,
        default="cohere/",
        help="Path to the directory with Parquet files",
    )
    arg_parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "int8"],
        help="Data type of the vectors",
    )
    arg_parser.add_argument(
        "--vector-column",
        type=str,
        default="emb",
        help="Name of the column with vectors in the Parquet files",
    )
    arg_parser.add_argument(
        "--metric",
        type=str,
        default="jaccard",
        choices=["jaccard", "hamming"],
        help="Metric to use for the index",
    )
    arg_parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of nearest neighbors to search for",
    )
    arg_parser.add_argument(
        "--ndim",
        type=int,
        default=None,
        help="Number of dimensions to use for the vectors",
    )
    arg_parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of vectors to load",
    )
    args = arg_parser.parse_args()
    main(
        data=args.data,
        dtype=args.dtype,
        vector_column=args.vector_column,
        metric=args.metric,
        k=args.k,
        ndim=args.ndim,
        limit=args.limit,
    )

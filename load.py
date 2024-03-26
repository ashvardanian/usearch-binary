import os
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def binarize(arr):
    arr = np.array(arr)
    mapped_arr = (arr >= 0).astype(np.uint8)
    packed_bits = np.packbits(mapped_arr)
    return packed_bits


def process_file(file_path: str, embeddings_column: str):
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return
    assert embeddings_column in df.columns, "Embeddings column missing"
    first_embedding_dims = len(df[embeddings_column][0])
    assert first_embedding_dims in [1024, 128], "Invalid embedding dimensions"
    if first_embedding_dims == 128:
        return
    df[embeddings_column] = df[embeddings_column].apply(binarize).apply(bytes)
    schema = pa.schema(
        [
            (embeddings_column, pa.binary(128)),
            ("_id", pa.string()),
            ("url", pa.string()),
            ("title", pa.string()),
            ("text", pa.string()),
        ]
    )
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, file_path)
    print(f"Processed and binarized {file_path}")


def download_file(file_url: str, local_file: str, embeddings_column: str):
    if os.path.exists(local_file):
        print(f"Skipping {local_file} download - already present")
        process_file(local_file, embeddings_column)
        return local_file
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    # Removed '-c' flag to prevent continuing from a potentially corrupted file
    subprocess.run(["wget", "-q", "-O", local_file, file_url], check=True)
    print(f"Downloaded {local_file}")
    process_file(local_file, embeddings_column)
    return local_file


def download_files(file_urls: list, local_files: list, embeddings_column: str = "emb"):
    workers = multiprocessing.cpu_count() - 2
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_url = {
            executor.submit(
                download_file,
                file_url,
                local_file,
                embeddings_column,
            ): file_url
            for file_url, local_file in zip(file_urls, local_files)
        }
        for future in as_completed(future_to_url):
            index = future_to_url[future]
            try:
                local_file = future.result()
                print(f"Finished processing {local_file}")
            except Exception as exc:
                print(f"File number {index} generated an exception: {exc}")


def download_mixedbread():
    base_url = "https://huggingface.co/datasets/mixedbread-ai/wikipedia-embed-en-2023-11/resolve/main/data/train-{index:05d}-of-00721.parquet?download=true"
    save_path = "mixedbread/{index:05d}.parquet"
    file_urls = [base_url.format(index=i) for i in range(721)]
    local_files = [save_path.format(index=i) for i in range(721)]
    download_files(file_urls, local_files)


def download_cohere(language: str = "en"):
    base_url = "https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3/resolve/main/{language}/{index:04d}.parquet?download=true"
    save_path = "cohere/{language}-{index:05d}.parquet"
    file_urls = [base_url.format(index=i, language=language) for i in range(415)]
    local_files = [save_path.format(index=i, language=language) for i in range(415)]
    download_files(file_urls, local_files)


if __name__ == "__main__":
    download_cohere()
    download_mixedbread()

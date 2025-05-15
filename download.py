#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "pyarrow",
#   "argparse",
# ]
# ///

import math
import os
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal
import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def binarize(arr) -> np.ndarray:
    arr = np.array(arr)
    mapped_arr = (arr >= 0).astype(np.uint8)
    packed_bits = np.packbits(mapped_arr)
    return packed_bits


def quantize_file(file_path: str, embeddings_column: str):
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
    # Log the number of rows processed, the binary combined size of quantized embeddings,
    # and the file size on disk:
    quantized_size = df[embeddings_column].apply(len).sum()
    file_size = os.path.getsize(file_path)
    print(f"Processed and binarized {file_path} with {len(df):,} rows")
    print(f"- Quantized vectors size: {quantized_size:,} bytes")
    print(f"- File size on disk: {file_size:,} bytes")


def download_file(
    file_url: str,
    local_file: str,
    embeddings_column: str,
    quantize: bool = False,
):
    if os.path.exists(local_file):
        print(f"Skipping {local_file} download - already present")
        if quantize:
            quantize_file(local_file, embeddings_column)
        return local_file
    os.makedirs(os.path.dirname(local_file), exist_ok=True)

    # Removed '-c' flag to prevent continuing from a potentially corrupted file
    subprocess.run(["wget", "-q", "-O", local_file, file_url], check=True)
    print(f"Downloaded {local_file}")
    if quantize:
        quantize_file(local_file, embeddings_column)
    return local_file


def download_files(
    file_urls: list,
    local_files: list,
    embeddings_column: str = "emb",
    quantize: bool = False,
):
    workers = multiprocessing.cpu_count() - 2
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_url = {
            executor.submit(
                download_file,
                file_url,
                local_file,
                embeddings_column=embeddings_column,
                quantize=quantize,
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


def download_mixedbread(**kwargs):
    base_url = "https://huggingface.co/datasets/mixedbread-ai/wikipedia-embed-en-2023-11/resolve/main/data/train-{index:05d}-of-00721.parquet?download=true"
    save_path = "mixedbread/{index:05d}.parquet"
    file_urls = [base_url.format(index=i) for i in range(721)]
    local_files = [save_path.format(index=i) for i in range(721)]
    download_files(file_urls, local_files, **kwargs)


def download_cohere(language: str = "en", count: int = 415, **kwargs):
    base_url = "https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3/resolve/main/{language}/{index:04d}.parquet?download=true"
    save_path = "cohere/{language}-{index:05d}.parquet"
    file_urls = [base_url.format(index=i, language=language) for i in range(count)]
    local_files = [save_path.format(index=i, language=language) for i in range(count)]
    download_files(file_urls, local_files, **kwargs)


def download_dataset(
    dataset_name: Literal["cohere-en", "mixedbread-en", "cohere"],
    quantize: bool = False,
):
    if dataset_name == "cohere-en":
        download_cohere("en", quantize=quantize)
    elif dataset_name == "mixedbread-en":
        download_mixedbread(quantize=quantize)
    elif dataset_name == "cohere":
        # Let's download all the other languages:
        languages = {
            "en": 41488110,
            "de": 20772081,
            "fr": 17813768,
            "ru": 13734543,
            "es": 12905284,
            "it": 10462162,
            "ceb": 9818657,
            "uk": 6901192,
            "ja": 6626537,
            "nl": 6101353,
            "pl": 5973650,
            "pt": 5637930,
            "sv": 4911480,
            "ca": 4156889,
            "ar": 3691451,
            "cs": 3118524,
            "he": 2948882,
            "hu": 2924609,
            "vi": 2835049,
            "zh": 2775260,
            "fi": 2427097,
            "id": 2358335,
            "no": 2211270,
            "sr": 2154631,
            "fa": 2073154,
            "tr": 1775036,
            "ro": 1770527,
            "el": 1599770,
            "ko": 1513291,
            "bg": 1455765,
            "hy": 1386140,
            "eu": 1327579,
            "da": 1224982,
            "eo": 1216706,
            "war": 1185097,
            "sh": 1139309,
            "tt": 1119546,
            "arz": 1089164,
            "gl": 1056990,
            "et": 1054770,
            "ce": 1013217,
            "ast": 1010445,
            "sl": 984855,
            "hr": 910923,
            "sk": 874014,
            "ms": 869579,
            "be": 857033,
            "th": 839712,
            "az": 817631,
            "uz": 811028,
            "mk": 784576,
            "lt": 770069,
            "bn": 767965,
            "cy": 762338,
            "ta": 685975,
            "simple": 646424,
            "te": 634778,
            "kk": 627085,
            "ka": 595401,
            "hi": 541822,
            "nn": 530590,
            "lv": 484957,
            "af": 461674,
            "ba": 434939,
            "ur": 434269,
            "bs": 396692,
            "sq": 388788,
            "ml": 384795,
            "min": 373156,
            "la": 340521,
            "pnb": 335958,
            "be-x-old": 314600,
            "kn": 309085,
            "azb": 294248,
            "oc": 283297,
            "zh-min-nan": 278547,
            "fy": 248075,
            "my": 241025,
            "lb": 216558,
            "ky": 216344,
            "als": 206387,
            "mr": 203479,
            "br": 200486,
            "pa": 188091,
            "is": 177272,
            "mg": 171947,
            "sw": 171650,
            "ha": 167807,
            "tl": 166907,
            "nds": 166019,
            "an": 143163,
            "jv": 142104,
            "ps": 138240,
            "ig": 132250,
            "new": 128696,
            "tg": 128237,
            "ga": 125456,
            "lld": 125094,
            "su": 124390,
            "cv": 122671,
            "ckb": 120886,
            "si": 119223,
            "mn": 114878,
            "lmo": 103836,
            "io": 101713,
            "gu": 99450,
            "vec": 95072,
            "zh-yue": 89145,
            "bar": 88238,
            "sco": 83906,
            "ne": 83598,
            "ku": 82935,
            "hyw": 82343,
            "pms": 77834,
            "as": 76093,
            "km": 74177,
            "sah": 71599,
            "li": 69267,
            "or": 65510,
            "mt": 64038,
            "szl": 56836,
            "yi": 55375,
            "ht": 55079,
            "dag": 53343,
            "sa": 51735,
            "nv": 49355,
            "bpy": 47757,
            "vo": 47375,
            "ug": 44764,
            "sat": 43500,
            "ia": 42012,
            "bo": 41438,
            "mwl": 41273,
            "sd": 40395,
            "bcl": 39967,
            "mnw": 39578,
            "hsb": 39560,
            "avk": 39001,
            "scn": 38359,
            "rm": 37436,
            "diq": 34743,
            "vep": 33654,
            "xmf": 33238,
            "ban": 32320,
            "wa": 32132,
            "ilo": 31046,
            "nds-nl": 30918,
            "qu": 30529,
            "so": 29936,
            "mhr": 29619,
            "vls": 29227,
            "sc": 28977,
            "fo": 28809,
            "gd": 28149,
            "rw": 28037,
            "gom": 27792,
            "yo": 27789,
            "tum": 26743,
            "wuu": 26532,
            "frr": 26010,
            "sn": 25941,
            "tk": 24269,
            "blk": 24194,
            "mzn": 23837,
            "co": 23065,
            "szy": 22854,
            "am": 22467,
            "shn": 22432,
            "skr": 21081,
            "lfn": 20781,
            "tyv": 20762,
            "lij": 20553,
            "ie": 19994,
            "rue": 19916,
            "crh": 19016,
            "gor": 18146,
            "ary": 17463,
            "dv": 16941,
            "lg": 16751,
            "roa-tara": 16572,
            "bjn": 16429,
            "tw": 16304,
            "bh": 15938,
            "pam": 15134,
            "os": 15096,
            "myv": 15062,
            "gn": 14983,
            "lez": 14152,
            "mai": 13806,
            "kv": 13534,
            "pcd": 13057,
            "zh-classical": 12791,
            "zea": 12528,
            "lo": 12525,
            "gv": 12074,
            "stq": 11890,
            "zu": 11680,
            "smn": 11672,
            "kw": 11539,
            "bat-smg": 11240,
            "hif": 11215,
            "ext": 10967,
            "ace": 10821,
            "trv": 10546,
            "ami": 10538,
            "tcy": 10531,
            "lad": 10386,
            "alt": 10256,
            "pap": 10187,
            "kab": 10179,
            "fur": 10148,
            "nap": 10079,
            "mrj": 9771,
            "kaa": 9548,
            "nqo": 9153,
            "glk": 9120,
            "pfl": 8790,
            "fiu-vro": 8757,
            "nso": 8635,
            "jbo": 8577,
            "bxr": 8549,
            "wo": 8549,
            "olo": 8530,
            "map-bms": 8393,
            "ksh": 8226,
            "csb": 8085,
            "av": 7873,
            "mni": 7740,
            "udm": 7730,
            "mi": 7643,
            "kbp": 7616,
            "dsb": 7536,
            "frp": 7294,
            "om": 7045,
            "ang": 7023,
            "hak": 6866,
            "gur": 6761,
            "se": 6733,
            "anp": 6704,
            "tay": 6434,
            "mdf": 6351,
            "gcr": 6347,
            "koi": 6300,
            "krc": 6293,
            "ay": 5985,
            "cdo": 5917,
            "nrm": 5786,
            "xh": 5756,
            "tn": 5712,
            "tly": 5598,
            "shi": 5179,
            "pcm": 5076,
            "fat": 4968,
            "nia": 4795,
            "dty": 4728,
            "kbd": 4667,
            "gpe": 4289,
            "cbk-zam": 4224,
            "ff": 4166,
            "dz": 4117,
            "guw": 3982,
            "eml": 3979,
            "ln": 3774,
            "inh": 3768,
            "nah": 3720,
            "ab": 3465,
            "ks": 3255,
            "mad": 3236,
            "haw": 3227,
            "gag": 3076,
            "tet": 3030,
            "ny": 2933,
            "pag": 2727,
            "guc": 2454,
            "roa-rup": 2409,
            "jam": 2387,
            "awa": 2242,
            "pdc": 2239,
            "to": 2165,
            "za": 2132,
            "st": 2051,
            "ltg": 2005,
            "atj": 1967,
            "nov": 1916,
            "ss": 1904,
            "pwn": 1881,
            "ee": 1819,
            "sm": 1659,
            "ts": 1645,
            "gan": 1626,
            "xal": 1619,
            "kcg": 1555,
            "cu": 1477,
            "srn": 1395,
            "got": 1280,
            "fon": 1247,
            "din": 1214,
            "arc": 1167,
            "fj": 1164,
            "rmy": 1113,
            "ady": 1040,
            "rn": 1033,
            "bm": 1017,
            "tpi": 957,
            "ve": 919,
            "ki": 798,
            "pnt": 796,
            "chr": 788,
            "kl": 770,
            "lbe": 766,
            "bi": 718,
            "ti": 706,
            "kg": 609,
            "pih": 606,
            "ch": 513,
            "bug": 429,
            "ty": 297,
            "ik": 275,
            "iu": 263,
            "pi": 260,
            "sg": 204,
            "chy": 57,
            "cr": 41,
        }

        rows_per_file = 100_000
        for language, docs in languages.items():
            expected_files = math.ceil(docs / rows_per_file)
            download_cohere(language, expected_files, quantize=quantize)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cohere-en", "mixedbread-en", "cohere"],
        default="cohere-en",
        help="Dataset to download",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the embeddings to individual bits",
    )
    args = parser.parse_args()
    download_dataset(args.dataset, quantize=args.quantize)

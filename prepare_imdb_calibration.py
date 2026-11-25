import pandas as pd
import numpy as np
import random
import requests

train = pd.read_parquet("data/raw/train-00000-of-00001.parquet")
test = pd.read_parquet("data/raw/test-00000-of-00001.parquet")

print(train.isna().sum())
print(train.value_counts('label' , dropna=False))

print(test.isna().sum())
print(test.value_counts('label' , dropna=False))

overlap = set(train["text"]) & set(test["text"])
print("Спільних унікальних текстів:", len(overlap))

is_overlap = train['text'].isin(overlap)
calibration_pool = train.loc[~is_overlap].copy()
print(calibration_pool.shape)

overlap = set(calibration_pool["text"]) & set(test["text"])
print("Спільних унікальних текстів:", len(overlap))

print(calibration_pool['text'].duplicated().sum())
calibration_pool = calibration_pool[~calibration_pool['text'].duplicated()].copy()

def tokenize_text(text):
    response = requests.post(
        "http://localhost:8080/tokenize",
        json={"content": text,
              "add_special": False,
              "parse_special": False,
              },
        timeout=60,
    )
    response.raise_for_status()

    tokens = response.json()["tokens"]
    return response.json()["tokens"]


reviews = calibration_pool["text"].tolist()

shuffle_reviews = reviews.copy()
random.Random(44).shuffle(shuffle_reviews)

calibration_text = "\n\n".join(shuffle_reviews[:1000])
tokens = tokenize_text(calibration_text)

print("Imbd total tokens:", len(tokens))
assert len(tokens) >= 65536, "Not enough tokens for calibration. Please check the dataset."

small_tokens = tokens[:16384]
large_tokens = tokens[:65536]

def detokenize_tokens(tokens):
    response = requests.post(
        "http://localhost:8080/detokenize",
        json={"tokens": tokens},
        timeout=60,
    )
    response.raise_for_status()

    return response.json()["content"]

small_text = detokenize_tokens(small_tokens)
large_text = detokenize_tokens(large_tokens)


from pathlib import Path

output_dir = Path(__file__).resolve().parent / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

small_path = output_dir / "calibration_imdb_seed44_16384.txt"
large_path = output_dir / "calibration_imdb_seed44_65536.txt"

small_path.write_text(small_text, encoding="utf-8")
large_path.write_text(large_text, encoding="utf-8")

print("Saved small:", len(tokenize_text(
    small_path.read_text(encoding="utf-8"))))
print("Saved large:", len(tokenize_text(
    large_path.read_text(encoding="utf-8"))))
from datasets import load_dataset



wiki = load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="train"
)
print(type(wiki))
print("Rows: ", len(wiki))

paragraphs = []
for i in range(len(wiki)):
    text = wiki[i]["text"].strip()
    if text ==  '' or text.startswith('='):
        continue
    paragraphs.append(text)
print("Paragraphs: ", len(paragraphs))

unique_paragraphs = list(dict.fromkeys(paragraphs))
print("Unique paragraphs: ", len(unique_paragraphs))
print("Duplicates: ", len(paragraphs) - len(unique_paragraphs))

import random

shuffle_paragraphs = unique_paragraphs.copy()
random.Random(44).shuffle(shuffle_paragraphs)

print("Shuffle paragraphs: ", shuffle_paragraphs[0])


import requests




def count_tokens(paragraph):

    response = requests.post(
        "http://localhost:8080/tokenize",
        json={"content": paragraph,
              "add_special": False,
              "parse_special": False,
              },
        timeout=60,
    )
    response.raise_for_status()

    tokens = response.json()["tokens"]
    return len(tokens)
print(count_tokens(shuffle_paragraphs[0]))

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

calibration_text = "\n\n".join(shuffle_paragraphs[:1000])
tokens = tokenize_text(calibration_text)

small_tokens = tokens[:16384]
large_tokens = tokens[:65536]

print("Total tokens:", len(tokens))
print("Small tokens:", len(small_tokens))
print("Large tokens:", len(large_tokens))


def detokenize_tokens(tokens):
    response = requests.post(
        "http://localhost:8080/detokenize",
        json={"tokens": tokens},
        timeout=60,
    )
    response.raise_for_status()

    text = response.json()["content"]
    return text

small_text = detokenize_tokens(small_tokens)
large_text = detokenize_tokens(large_tokens)

print("Small text tokens:", count_tokens(small_text))
print("Large text tokens:", count_tokens(large_text))

from pathlib import Path

output_dir = Path(__file__).parent.resolve() / "data/processed"
output_dir.mkdir(parents=True, exist_ok=True)

small_path = output_dir / "calibration_wikitext_seed44_16384.txt"
large_path = output_dir / "calibration_wikitext_seed44_65536.txt"

small_path.write_text(small_text, encoding="utf-8")
large_path.write_text(large_text, encoding="utf-8")

print("Saved small:", count_tokens(small_path.read_text(encoding="utf-8")))
print("Saved large:", count_tokens(large_path.read_text(encoding="utf-8")))
from pathlib import Path
import re

import pandas as pd


data_dir = Path(__file__).resolve().parent / "data" / "processed"
test = pd.read_csv(data_dir / "main_test_1000.csv")
assert len(test) == 1000 and test["source_row_id"].is_unique

pattern = re.compile(
    r"q4_k_m_(imdb|wikitext)_seed(42|43|44)_(16384|65536)_predictions\.csv"
)
label_to_id = {"NEGATIVE": 0, "POSITIVE": 1}
rows = []

for path in sorted(data_dir.glob("*_predictions.csv")):
    match = pattern.fullmatch(path.name)
    if match is None:
        continue

    source, seed, tokens = match.groups()
    result = pd.read_csv(path)

    assert len(result) == len(test), f"{path.name}: неправильна кількість рядків"
    assert result["row_id"].equals(test["source_row_id"]), (
        f"{path.name}: ідентифікатори або порядок відгуків відрізняються"
    )
    assert result["actual_label"].equals(test["label"]), (
        f"{path.name}: правильні мітки відрізняються"
    )

    predicted = result["predicted_label"].str.strip().map(label_to_id)
    correct = predicted.eq(result["actual_label"]).sum()

    rows.append({
        "source": source,
        "tokens": int(tokens),
        "seed": int(seed),
        "correct": correct,
        "accuracy_percent": 100 * correct / len(test),
        "unrecognized": predicted.isna().sum(),
    })

summary = pd.DataFrame(rows)
assert len(summary) == 12, f"Знайдено {len(summary)} із 12 очікуваних CSV"

print(summary.sort_values(["source", "tokens", "seed"]).to_string(index=False))

print("\nПідсумки за джерелом і обсягом")
print(
    summary.groupby(["source", "tokens"])["accuracy_percent"]
    .agg(["mean", "min", "max"])
    .round(2)
)
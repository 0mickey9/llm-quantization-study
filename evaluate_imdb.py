from pathlib import Path
import pandas as pd



data_dir = Path(__file__).resolve().parent / "data"
test_path = data_dir / "processed" / "main_test_1000.csv"

if not test_path.exists():
    test = pd.read_parquet(
        data_dir / "raw" / "test-00000-of-00001.parquet"
    )

    test = test.drop_duplicates(subset=["text"]).copy()
    test["source_row_id"] = test.index

    main_test = test.groupby("label").sample(
        n=500,
        random_state=42,
    )

    test_path.parent.mkdir(parents=True, exist_ok=True)
    main_test.to_csv(test_path, index=False)

ok = pd.read_csv(test_path)

assert len(ok) == 1000
assert ok["text"].notna().all()
assert ok["text"].is_unique
assert ok["source_row_id"].is_unique
assert ok["label"].value_counts().to_dict() == {0: 500, 1: 500}

print("Test shape:", ok.shape)
print(ok["label"].value_counts())

import requests

run_name = "q4_k_m_no_imatrix_main"
predictions_path = data_dir / "processed" / f"{run_name}_predictions.csv"

if predictions_path.exists():
    raise FileExistsError(f"Результати вже існують: {predictions_path}")

results = []
for i in range(len(ok)):
    review = ok['text'].iloc[i]
    prompt = f"""/no_think
    Classify the overall sentiment of the movie review below.
    Return exactly one word: POSITIVE or NEGATIVE.
    <review>
    {review}
    </review>"""
    payload = {
        "messages" : [{ "role" : "user", "content" : prompt}],
        "temperature" : 0,
        "seed" : 42,
    }
    response = requests.post("http://127.0.0.1:8080/v1/chat/completions", json  = payload, timeout = 60,)
    response.raise_for_status()

    prediction = response.json()["choices"][0]['message']['content'].strip()
    if (i + 1) % 50 == 0:
        print(f"Processed: {i + 1}/{len(ok)}")

    results.append(
        {
            "row_id" : int(ok["source_row_id"].iloc[i]),
            "predicted_label" : prediction,
            "actual_label" : ok["label"].iloc[i]
        }
    )
print(len(results))

result = pd.DataFrame(results)
result.to_csv(predictions_path, index=False)
print("Saved to:", predictions_path)




predicted_ids = result["predicted_label"].map({"NEGATIVE" : 0, "POSITIVE" : 1})
correct = predicted_ids.eq(result["actual_label"]).fillna(False)
print("Correct / total:", correct.sum(), "/", len(result))
print("Accuracy:", f"{correct.mean():.2%}")
print("Unrecognized labels:", predicted_ids.isna().sum())

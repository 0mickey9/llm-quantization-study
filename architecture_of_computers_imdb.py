import pandas as pd


# print(pd.__version__)
#
# data = pd.read_csv(r'C:\Users\Admin\OneDrive\Desktop\for school\Architecture of computers\Projects\data\raw\IMDB Dataset.csv')
# print(data.shape)
# print(data.head())
#
# print(data.isna().sum())
# print(data.value_counts('sentiment'))
#
# print(data.duplicated().sum())
# print(data.duplicated(subset = ['review']).sum())
#
# print(data['review'].str.strip().eq('').sum())
#
# clean_data = data.drop_duplicates(subset = ['review']).copy()
# print(clean_data.shape)
#
# print(clean_data.value_counts('sentiment'))
# print(clean_data['review'].str.contains("<br />", regex = False).sum())
# clean_data['review'] = clean_data['review'].str.replace("<br />", ' ', regex = False)
# print(clean_data["review"].str.contains(r"<br\s*/?>", regex=True).sum())
# print(clean_data.duplicated(subset=["review"]).sum())
#
# clean_data['review'] = clean_data['review'].str.replace(r'\s+', ' ', regex = True).str.strip()
# len_of_rows = clean_data["review"].str.len()
# print(len_of_rows.describe())
#
#
# from pathlib import Path
#
# output_path = Path(__file__).resolve().parent / "data" / "processed" / "cleaned_IMDB_Dataset.csv"
# output_path.parent.mkdir(parents = True, exist_ok = True)
# clean_data.to_csv(output_path, index=False)
# saved_data = pd.read_csv(output_path)
# print(saved_data.shape)
#
# print(clean_data.reset_index(drop=True).equals(
#     saved_data.reset_index(drop = True)
#  )
# )
with_matrix = pd.read_csv("data/processed/imdb_q4_k_m_imdb_pilot_predictions.csv")
without_matrix = pd.read_csv("data/processed/imdb_q4_k_m_no_imatrix_predictions.csv")

assert with_matrix["row_id"].equals(without_matrix["row_id"])
assert with_matrix["actual_label"].equals(without_matrix["actual_label"])

differences = with_matrix["predicted_label"].ne(without_matrix["predicted_label"])
print("Different predictions:", differences.sum())


from pathlib import Path
import requests

text_path = Path(__file__).resolve().parent / "data/processed/calibration_imdb_pilot.txt"
text = text_path.read_text(encoding = 'utf-8')

response = requests.post(
    "http://127.0.0.1:8080/tokenize",
    json = {"content" : text,
            "add_specials" : False,
            "parse_specials" : False,
            },
    timeout = 60,
)
response.raise_for_status()

tokens = response.json()['tokens']
print("Calibrations tokens : ", len(tokens))

raise SystemExit


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
print(calibration_pool.shape)
print(calibration_pool["text"].duplicated().sum())

calibration_reviews = calibration_pool.sample(n = 100, random_state= 42)
calibration_text = "\n\n".join(calibration_reviews.text)

pilot_test = test.groupby('label').sample(n = 100, random_state = 42)
print(pilot_test.shape)
print(pilot_test.value_counts('label', dropna = False), '\n')

pilot_test.to_csv("data/processed/pilot_test.csv", index = False)
ok = pd.read_csv("data/processed/pilot_test.csv")
positive_reviews = ok[ok["label"] == 1]
print(positive_reviews["text"].iloc[0])
print(ok.shape)
print(ok.iloc[0])

import requests

reveiw = ok["text"].iloc[0]
prompt = f"""/no_think
    Classify the overall sentiment of the movie review below.
    Return exactly one word: POSITIVE or NEGATIVE.
    <review>
    {reveiw}
    </review>"""
payload = {
    "messages" : [{"role" : "user", "content" : prompt}],
    "temperature" : 0,
    "seed" : 42,
}
response = requests.post("http://127.0.0.1:8080/v1/chat/completions", json = payload, timeout = 60 ,)
response.raise_for_status()

prediction = response.json()["choices"][0]["message"]["content"].strip()
print("Predicted label:", prediction)

# launching llama.cpp to classify the review


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
    print()
    print("Predicted label:", prediction)
    print(ok["label"].iloc[i])

    results.append(
        {
            "row_id" : i,
            "predicted_label" : prediction,
            "actual_label" : ok["label"].iloc[i]
        }
    )
print(len(results))

result = pd.DataFrame(results)
result.to_csv(
    "data/processed/imdb_q4_k_m_no_imatrix_predictions.csv",
    index=False,
)
print(result.shape)




predicted_ids = result["predicted_label"].map({"NEGATIVE" : 0, "POSITIVE" : 1})
correct = predicted_ids.eq(result["actual_label"]).fillna(False)
print("Correct / total:", correct.sum(), "/", len(result))
print("Accuracy:", f"{correct.mean():.2%}")
print("Unrecognized labels:", predicted_ids.isna().sum())

from pathlib import Path

calibration_path = (
    Path(__file__).resolve().parent
    / "data/processed/calibration_imdb_pilot.txt"
)
calibration_path.write_text(calibration_text, encoding = "utf-8")

print("Saved to : " , calibration_path)
raise SystemExit



from pathlib import Path
import subprocess
llama_dir = Path(r"C:\Users\Admin\llama.cpp")
prompt_path = llama_dir / "imdb_pilot_prompt.txt"
asnwers = []
for i in range(5):
    print(f"Iteration {i+1}")
    review = ok["text"].iloc[i]
    prompt = f"""/no_think
        Classify the overall sentiment of the movie review below.
        Return exactly one word: POSITIVE or NEGATIVE.

        <review>
        {review}
        </review>"""

    prompt_path.write_text(prompt, encoding="utf-8")

    results = subprocess.run(
        [
            str(llama_dir / "llama-cli.exe"),
            "-m", r".\models\Qwen3-1.7B-f16.gguf",
            "-ngl", "99",
            "-c", "4096",
            "--temp", "0",
            "--seed", "42",
            "--single-turn",
            "-f", str(prompt_path),
        ],
        cwd=llama_dir,
        check=True,
        capture_output = True,
        encoding="utf-8",
        text = True
    )

    print(results.stdout)

    labels = [
        line.strip() for line in results.stdout.splitlines()
        if line.strip() in ["POSITIVE", "NEGATIVE"]
    ]

    prediction = labels[0] if len(labels) == 1 else None
    print("Predicted label:", prediction)


    label_to_id = {"NEGATIVE" : 0, "POSITIVE" : 1}
    predicted_id = label_to_id.get(prediction)
    actual_id = ok['label'].iloc[i]

    asnwers.append((predicted_id, actual_id))

for i in range(len(asnwers)):
    predicted_id, actual_id = asnwers[i]
    print(f"Review {i+1}: \\n")
    print("Predicted IMDb label:", predicted_id)
    print("Actual IMDb label:", actual_id)
    print("Correct:", predicted_id == actual_id)
# Effects of Calibration Data Source and Size on Quantized Model Accuracy

This project tests how calibration data affects the accuracy of Qwen3-1.7B quantized in the Q4_K_M format. It compares IMDb and WikiText calibration texts at 16,384 and 65,536 tokens, with three sampling seeds per condition. All models were evaluated on the same 1,000 IMDb reviews, with 500 positive and 500 negative reviews.

## Results

| Calibration source | Tokens | Seed 42 | Seed 43 | Seed 44 | Mean accuracy |
|---|---:|---:|---:|---:|---:|
| IMDb | 16,384 | 91.90% | 91.20% | 87.10% | 90.07% |
| IMDb | 65,536 | 91.50% | 91.20% | 90.10% | 90.93% |
| WikiText | 16,384 | 92.20% | 92.30% | 92.00% | 92.17% |
| WikiText | 65,536 | 91.50% | 91.90% | 92.30% | 91.90% |

The F16 reference scored **92.00% (920/1,000)**. The Q4_K_M baseline without an importance matrix scored **87.50% (875/1,000)**.

## Data and evidence

- [Main test set](data/processed/main_test_1000.csv)
- [Calibration texts and predictions](data/processed/)
- [F16 predictions](data/processed/f16_main_predictions.csv)
- [Q4_K_M baseline predictions](data/processed/q4_k_m_no_imatrix_main_predictions.csv)
- [Raw data](data/raw/)
- Upstream datasets: [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) and [WikiText](https://huggingface.co/datasets/Salesforce/wikitext), configuration `wikitext-2-raw-v1`.

The 12 main calibration runs use the filename pattern `q4_k_m_<source>_seed<seed>_<tokens>_predictions.csv`. Files marked `pilot` are from preliminary runs.

## Verify the results from saved predictions

From the repository root, run:

```sh
python -m pip install pandas
python check_results.py
```

The script checks the 12 calibrated runs against the review IDs and labels in the main test set. It reports accuracy, out-of-format responses, and group means and ranges. After trimming surrounding whitespace, only exact `POSITIVE` and `NEGATIVE` responses can count as correct. Both baseline CSVs can be scored using the same rule. They are separate from the 12-run summary.

## Rerun the experiment

You need Python, an F16 GGUF of [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF), and [llama.cpp](https://github.com/ggml-org/llama.cpp) with `llama-server`, `llama-imatrix`, and `llama-quantize` available on your PATH. The examples assume the input model is named `Qwen3-1.7B-f16.gguf`. Adjust paths and GPU offload for your machine.

### 1. Prepare calibration texts

Use the committed calibration texts to rerun the experiment with the same inputs. To regenerate the texts, install the Python dependencies and start `llama-server` with the F16 model:

```sh
python -m pip install pandas numpy pyarrow datasets requests
llama-server -m Qwen3-1.7B-f16.gguf -ngl 99 -c 4096 --host 127.0.0.1 --port 8080
```

In a separate terminal, run `python prepare_imdb_calibration.py` and `python prepare_wikitest.py` from the repository root. Both scripts use seed 44 as committed. For seeds 42 and 43, update the `random.Random` seed and the seed in both output filenames before each run. The WikiText script loads the dataset from Hugging Face. Regenerate files in a separate checkout to preserve the saved evidence.

### 2. Create importance matrices and quantize

The following example uses WikiText with seed 43 and 16,384 tokens:

```sh
llama-imatrix -m Qwen3-1.7B-f16.gguf -f data/processed/calibration_wikitext_seed43_16384.txt -o imatrix_wikitext_seed43_16384.gguf -c 512 -b 2048 -ngl 99
llama-quantize --imatrix imatrix_wikitext_seed43_16384.gguf Qwen3-1.7B-f16.gguf Qwen3-1.7B-Q4_K_M-wikitext-seed43-16384.gguf Q4_K_M
```

Repeat these commands for each of the 12 calibration files, with unique matrix and output model names. Create the baseline without an importance matrix using:

```sh
llama-quantize Qwen3-1.7B-f16.gguf Qwen3-1.7B-Q4_K_M-no-imatrix.gguf Q4_K_M
```

### 3. Evaluate each model

Start a fresh server for each model. For the WikiText example above:

```sh
llama-server -m Qwen3-1.7B-Q4_K_M-wikitext-seed43-16384.gguf -ngl 99 -c 4096 --host 127.0.0.1 --port 8080
```

In [evaluate_imdb.py](evaluate_imdb.py), set `run_name` to match the loaded model. For the example, use `q4_k_m_wikitext_seed43_16384`. Use `f16_main` and `q4_k_m_no_imatrix_main` for the baselines. The script uses the committed test set, temperature 0, and generation seed 42.

Run `python evaluate_imdb.py` for each model. The script refuses to overwrite existing predictions, so move each original prediction CSV aside in your separate checkout before rerunning that model. Once all runs are complete, run `python check_results.py` to recompute the comparison.

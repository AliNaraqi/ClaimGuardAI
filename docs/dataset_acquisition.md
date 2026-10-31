# Dataset Acquisition

## Source

**Vehicle Insurance Claim Fraud Detection** — Kaggle, published by user `shivamb`.
https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection

## Steps

1. Create a free Kaggle account if you don't have one, and sign in.
2. Visit the dataset page above and accept any usage terms shown on the page.
3. Download the dataset (via the web UI "Download" button, or the Kaggle CLI: `kaggle datasets download -d shivamb/vehicle-claim-fraud-detection`).
4. Unzip it and locate `fraud_oracle.csv`.
5. Place the file at `data/raw/fraud_oracle.csv` in this repository (this path is gitignored — the file is never committed).

## Verification

After placing the file, confirm it matches what this project's notebooks/tests expect:

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/raw/fraud_oracle.csv')
assert df.shape == (15420, 33), df.shape
assert 'FraudFound_P' in df.columns
print('OK:', df.shape)
"
```

If the shape or columns differ, the dataset on Kaggle may have been updated since this project was built (last verified: October 2026) — re-check `docs/data_dictionary.md` against the new file before proceeding, and do not assume downstream notebooks will still work unmodified.

## Why it isn't committed to this repository

1. **Licensing.** The dataset's Kaggle terms require checking redistribution rights before re-publishing the file; this project does not assume that permission.
2. **Reproducibility hygiene.** Keeping raw third-party data out of version control is standard practice (`data/raw/` is listed in `.gitignore`) — it keeps the repo small and makes the acquisition step explicit and auditable rather than silently bundled.

## Dependency reproducibility

All dependency versions are pinned in [`requirements.txt`](../requirements.txt) (not just lower-bounded), including two versions pinned specifically to avoid known local build/compatibility issues encountered while building this project:

- `llvmlite==0.43.0` / `numba==0.60.0` — newer `llvmlite` has no prebuilt wheel for this environment's Python/macOS combination and fails to build from source.
- `pyarrow==17.0.0` — newer `pyarrow` requires NumPy ≥2.0, which conflicts with the NumPy version XGBoost/SHAP need here; without this pin, every Streamlit chart/dataframe call fails with an `ImportError` at runtime.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

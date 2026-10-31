# Model Card — ClaimGuard AI Risk Model

## Model details

- **Type:** Gradient-boosted tree classifier (XGBoost, 300 trees, max_depth=4, learning_rate=0.05)
- **Selected over:** a regularized Logistic Regression baseline (beaten on every validation metric — see [`docs/model_evaluation.md`](model_evaluation.md))
- **Version:** `xgboost_week2_baseline` (trained in [`notebooks/02_baseline_modeling.ipynb`](../notebooks/02_baseline_modeling.ipynb), `random_state=42`)
- **Inputs:** 29 policy/incident/claim fields (see [`docs/data_dictionary.md`](data_dictionary.md)); `PolicyNumber` and `RepNumber` are excluded as non-predictive identifiers
- **Output:** a single risk score in [0, 1] — a ranking signal, **not** a calibrated probability (see Limitations)
- **Companion signal:** an independent Isolation Forest anomaly score (unsupervised, never sees the target) — see [`docs/explainability_and_anomaly.md`](explainability_and_anomaly.md)

## Intended use

- **Primary use:** help an insurance claims analyst **prioritize a review queue** of historical/demo vehicle insurance claims by estimated risk, with per-claim SHAP explanations to support (not replace) human judgment.
- **Out of scope / prohibited:** automated claim denial, any individual fraud accusation, eligibility determination, or use as evidence in an investigation. This is a **portfolio/research demonstration**, not a production underwriting or fraud system.

## Training data

- [Vehicle Insurance Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) (Kaggle), 15,420 claims, 1994–1996.
- Target `FraudFound_P`: a dataset-provided historical flag of unknown adjudication process — **not** a verified legal fraud finding. 5.99% positive prevalence.
- Full data-quality findings (sentinel values, the `Age==0` quirk, leakage audit) in [`docs/eda_findings.md`](eda_findings.md) and [`docs/data_dictionary.md`](data_dictionary.md).

## Evaluation results (held-out test set, scored once)

| Metric | Value |
|---|---|
| ROC-AUC | 0.833 |
| PR-AUC | 0.242 |
| Precision / Recall @ 0.5 | 0.163 / 0.741 |
| Brier score | 0.134 |
| Recall @ top 20% reviewed | 61.1% |

Full validation comparison, curves, and confusion matrix in [`docs/model_evaluation.md`](model_evaluation.md). All numbers are copied directly from `reports/model_results.json`, produced by the notebook — never hand-typed or estimated.

## Limitations

1. **Historical, narrow scope.** 1994–1996 claims only. Results describe this dataset, not current insurance fraud patterns, pricing, or demographics.
2. **Label uncertainty.** `FraudFound_P` reflects whatever the original insurer's process flagged — not a confirmed, adjudicated outcome. Every score inherits this uncertainty.
3. **Not calibrated.** The calibration curve shows the model is overconfident (predicted 60–80% maps to an observed ~15–23% rate). Use only as a ranking signal.
4. **Explainability instability.** SHAP global/local importance is partly dominated by a few rare `Make` categories (VW, Dodge, Nisson) in a way that looks like an artifact of sparse one-hot tree splits, not a stable signal — see [`docs/explainability_and_anomaly.md`](explainability_and_anomaly.md) §1. Treat `Make`-based explanations with caution until addressed.
5. **No fairness audit yet.** `Sex` and `MaritalStatus` are included as model inputs. A disparate-impact / group-fairness audit across these attributes has **not** been performed — required before any claim about fairness or production-readiness.
6. **Seed stability: checked, acceptable spread.** Retrained/re-evaluated across 5 random seeds — PR-AUC ranged 0.242–0.265 (std ≈ 0.01 on a mean of ≈0.25). The reported headline metric is at the low end of this range, not a favorable outlier. Full cross-validation (vs. single splits per seed) has not been done.
7. **Class imbalance.** At ~6% positive prevalence, a fixed 0.5 threshold produces a high false-positive load (per claim detail in `docs/model_evaluation.md`) — any deployment would need a review-budget-based threshold (precision/recall@K), not the default cutoff.

## Ethical considerations

- Never present this model's output as evidence of wrongdoing, a legal finding, or a basis for automated denial — every page of the dashboard and the API response carries this disclaimer.
- Protected attributes (`Sex`, `MaritalStatus`, age) are present in training data; their inclusion is a judgment call made for this baseline, not a settled design decision — see Limitation 5.
- No real customer data is used anywhere in this project; the dashboard/API operate only on the public historical dataset or user-supplied synthetic/demo values.

## How to reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# download fraud_oracle.csv from Kaggle into data/raw/
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline_modeling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_explainability_and_anomaly.ipynb
pytest tests/ -v
streamlit run app/dashboard/app.py
```

# Week 1 EDA Findings — ClaimGuard AI

**Dataset:** `fraud_oracle.csv`, 15,420 rows × 33 columns, claims from 1994–1996
**Target:** `FraudFound_P`, 923 positive / 14,497 negative (**5.99% positive prevalence**)
**Full reproducible analysis:** [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb)
**Data dictionary:** [`docs/data_dictionary.md`](data_dictionary.md)

## Summary

The dataset is clean in the narrow sense (no true nulls, no duplicate rows) but has two sentinel-encoded quality issues that a naive `.isna()` check would miss: a `'0'` placeholder in `DayOfWeekClaimed`/`MonthClaimed` (1 row) and an `Age == 0` placeholder that exactly corresponds to the `AgeOfPolicyHolder == '16 to 17'` bucket (320 rows). Both are documented and will be handled explicitly in preprocessing rather than fed to a model as literal values.

The target is heavily imbalanced (~6% positive), so Week 2 evaluation will center on PR-AUC, precision/recall@K, and calibration rather than accuracy, with class weighting or resampling applied strictly inside training folds.

No column representing a post-investigation outcome (e.g., confirmed fraud recovery, investigation result) exists in the data, so the main leakage risk is **identifier leakage** — `PolicyNumber` is a unique-per-row ID and `RepNumber` is a claims-rep ID, neither of which is a legitimate predictive signal and both will be excluded from the feature set. `Fault` is flagged for documentation (fault may not be fully finalized at the moment of first scoring) but is not excluded, since the dataset provides no alternative "fault at intake" field.

## Descriptive associations (not causal, not fraud evidence)

Within this historical, dataset-defined label:
- Claims where `Fault == 'Policy Holder'` show a markedly higher `FraudFound_P` rate (~7.9%) than `Third Party`-fault claims (~0.9%).
- `BasePolicy == 'All Perils'` claims flag at ~10.1% vs. `Liability` at ~0.7%.
- `VehicleCategory == 'Utility'` flags highest (~11.2%), `Sport` lowest (~1.5%).
- Flag rate decreases as `PastNumberOfClaims` increases (highest for policyholders with no prior claims, ~7.8%) — counter to a naive "repeat claimants are riskier" assumption, and a reminder to treat every association as a property of this specific dataset's labeling process, not a general insurance-fraud finding.

These patterns likely reflect a mix of real behavioral signal and confounding with policy type/coverage selection; they are reported descriptively to inform feature engineering, not as validated risk factors.

## Caveats and limitations (carried into every later artifact)

1. **Historical scope:** all claims are from 1994–1996. Findings describe that era only.
2. **Label uncertainty:** `FraudFound_P` is whatever the original insurer's process flagged — not a confirmed, adjudicated fraud outcome. Treat as a dataset-defined proxy label throughout.
3. **Class imbalance:** 5.99% positive rate requires imbalance-aware modeling and evaluation (PR-AUC, precision/recall@K, calibration) rather than accuracy.
4. **Sentinel-encoded missingness:** `Age == 0` (320 rows) and the single `'0'`-coded claimed-date row must be explicitly handled, not treated as literal values.
5. **Identifiers excluded:** `PolicyNumber`, `RepNumber` are not predictive features and will be dropped before training.
6. **Protected attributes present:** `Sex`, `MaritalStatus`, age fields — requires fairness consideration per the project's Responsible AI guidance before/if included in the model.
7. **No GitHub repo, model, or dashboard has been built or published yet** — this document reflects Week 1 (data validation + EDA) only.

## Week 1 exit-gate check

> *"You can explain what every included model feature represents, and whether it would realistically be available when an analyst scores a claim."*

Addressed in full in the data dictionary's "Scoring-time available" column and the notebook's Section 7 leakage-audit table. Status: **met**, pending final feature-list sign-off when Week 2 preprocessing pipelines are built.

# ClaimGuard AI — Full Project Specification & Roadmap

**Project:** Explainable Insurance Claims Risk & Decision Intelligence Platform   
**Target timeline:** 4 weeks
**Primary audience:** Data Scientist, Data Analyst, Applied ML Engineer, Risk Analytics and Insurance Analytics recruiters  
**Deliverable:** Public GitHub repository with reproducible ML experiments, an interactive dashboard, documented model limitations and a demo  
**Status:** Planning specification — not a claim that the system or trained models are complete.

## 1. Executive summary

ClaimGuard AI is an analyst-facing insurance-claims decision-support application. It helps analysts examine historical claim patterns, estimate the relative probability of a *dataset-defined* suspicious/fraud-reported claim, flag statistically anomalous claims, explain model outputs, and prioritize a queue for human review.

**What the model does not do:** establish that a person committed fraud, automatically reject claims, determine eligibility for insurance, or make real decisions about individuals. Public datasets are historical and have uncertain provenance and potentially biased labels. This is a portfolio and research demonstration.

### Portfolio goals

- Demonstrate end-to-end data handling and SQL-based analysis.
- Compare interpretable ML baselines with tree ensembles.
- Handle class imbalance and probability calibration.
- Include global and local SHAP explanations, as well as outlier detection.
- Deliver a functional Streamlit interface, with a FastAPI endpoint if time permits.
- Showcase software engineering: reproducibility, unit tests, CI, documentation, Docker and a live demo.
- Communicate model limitations, dataset provenance, fairness risks and appropriate human oversight.

## 2. Exact public dataset links

### A. Primary choice: Vehicle Insurance Claim Fraud Detection (Kaggle)

**Link:** https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection

- **Typical file:** `fraud_oracle.csv`.
- **Reported size:** 15,420 rows and 33 columns; verify after downloading.
- **Target:** `FraudFound_P` (binary fraud flag as provided in the source).
- **Data type:** historical vehicle insurance claim, policy and incident fields.
- **Use:** supervised binary classification, class-imbalance assessment, risk ranking, investigation dashboard and explanations.
- **Caution:** older dataset; inspect provenance, usage rights, sensitive attributes, time-related leakage, and whether labels actually indicate confirmed fraud. Do not interpret the risk scores as calibrated for a current insurer.


**Recommendation:** Make this the single primary supervised dataset for MVP. Its larger size makes a more useful portfolio demonstration than a 1,000-row alternative, though it is still not representative of modern claims.

### B. Alternative classification dataset: Auto Insurance Claims Data (Kaggle)

**Link:** https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data

- **Typical file:** `insurance_claims.csv`.
- **Reported size:** approximately 1,000 records and 40 columns; verify after downloading.
- **Target:** `fraud_reported`, conventionally `Y` / `N`.
- **Use:** quick proof-of-concept, exploratory analysis, model explainability and feature-engineering experiments.
- **Tradeoff:** limited sample size, uncertain representativeness, and high likelihood of unstable performance estimates. Avoid large accuracy claims and use cross-validation.
- **Access and rights:** review Kaggle dataset page and license before redistribution.

### C. Optional claims-severity dataset: Allstate Claims Severity (Kaggle)

**Link:** https://www.kaggle.com/competitions/allstate-claims-severity/data

- **Target:** `loss`, a numeric claims-loss outcome.
- **Task:** regression, **not fraud classification**.
- **Use:** a later, clearly separated second module that forecasts claim severity / loss.
- **Access:** competition files may require Kaggle sign-in and agreement to competition terms; do not assume redistribution is allowed.

### D. Optional adjacent dataset: CFPB Consumer Complaint Database

**Link:** https://www.consumerfinance.gov/data-research/consumer-complaints/

- **Task:** complaint volume, trend, product and company-response analytics, **not insurance fraud detection**.
- **Use:** only for an optional separate complaint-intelligence experiment; do not combine its labels with claims fraud.
- **2026 consideration:** September 2026 API release notes report removal of consumer complaint narratives and related visualizations, so verify fields available before planning NLP on narrative text.
- **API release notes:** https://cfpb.github.io/api/ccdb/release-notes.html

### E. Synthetic data option (backup, not equivalent to evidence)

If Kaggle access or reuse conditions block the MVP, create a reproducible generator with a documented random seed and generate **5,000–20,000 fictional claims**. Suggested fields: claim identifier, claim type, claim amount, estimated coverage, policy tenure, prior-claim count, incident/reporting delay, number of attachments, incident category, location category, and `review_required`.

Design labels probabilistically, with transparent assumptions and randomness. Explicitly label `review_required` as **simulated review need**, not fraud. Keep train, test and generator assumptions documented. High performance on synthetic labels mainly demonstrates the model can recover assumptions designed into the generator, **not** that it detects real suspicious claims.

### Dataset decision

**Start with A (Vehicle Insurance Claim Fraud Detection).** If inaccessible, use B for a smaller demonstration. Use E only if a usable public dataset is not available. Save the original file locally under `data/raw/` and document download instructions; do not commit third-party data without permission. Do not combine fraud labels from A and B without justifying their different definitions and schemas.

## 3. Business problem and success criteria

### User persona

A claims analyst needs to triage a queue of incoming claims and understand which unusual cases deserve a closer human look, without treating an automated score as evidence of wrongdoing.

### Questions the system should answer

1. How many cases are in the analysis window, and what proportion carry the dataset-provided flag?
2. Which operational characteristics are associated with that flag?
3. How reliably can models rank cases on held-out data?
4. How many flagged cases would appear among the highest-ranked 5%, 10%, or 20%?
5. Why did the model give a particular case its score?
6. Which cases look atypical even without a known positive label?
7. How do model predictions change under different review thresholds?

### MVP acceptance criteria

- Reproduce all analysis from documented source-data acquisition steps.
- Validate schema, types, missingness and duplicate handling.
- Train an interpretable baseline and at least one tree-based model.
- Present held-out PR-AUC, precision, recall, F1, ROC-AUC and confusion matrix; report positive-class prevalence.
- Display precision@K and recall@K for a stated review budget.
- Include calibration plot or Brier score, with any calibration method fitted only using training/validation data.
- Provide at least one global and one per-claim model explanation.
- Allow users to explore the claim queue, filter by risk score and export selected results.
- Show prominent warnings about dataset limitations and human review.
- Publish README, requirements, tests and screenshots; ideally provide a live demo.

## 4. Feature scope

### Must-have (v1.0)

**Data ingestion and quality**
- Load an approved CSV; validate expected columns and target values.
- Show missing-value distribution, target prevalence, duplicates and datatype checks.
- Create a data dictionary; remove or isolate identifiers that are not legitimate predictors.

**Descriptive analytics**
- Number of claims; total/average claim amount where available; flag prevalence.
- Claim trends and breakdowns only when timestamps and field meanings support them.
- Distribution by incident/claim type and selected operational factors.

**Risk modeling**
- Baseline: regularized Logistic Regression.
- Candidate: XGBoost or HistGradientBoosting / Random Forest.
- Train-only preprocessing, encoding and imputation inside reusable pipelines.
- Class weights or other carefully validated strategies for minority class.
- Track model versions, input features, chosen threshold and metrics.

**Anomaly detection**
- Isolation Forest on appropriately preprocessed numerical/encoded features.
- Treat anomaly score as a separate signal, not a fraud probability.
- Visually distinguish model-ranked risk from statistical outliers.

**Explainability**
- Global feature importance and SHAP summary.
- Per-claim top contributing features with the model version and baseline clearly identified.
- Do not claim SHAP identifies causes; it explains a model prediction.

**Dashboard**
- Overview page, dataset-quality page, model-performance page and investigation queue.
- Filters; individual-claim summary; explanation; CSV export.
- Demo mode with safe sample cases.

### Nice-to-have after MVP

- FastAPI scoring and batch-prediction endpoints with input-schema validation.
- PostgreSQL or DuckDB-backed query layer and SQL views.
- MLflow experiment tracking; model monitoring and drift checks.
- Docker deploy, automated CI and a hosted demo.
- PDF report export and an LLM-generated *draft* analyst summary grounded exclusively in approved structured evidence, with review required.

**Do not make a generative AI agent a dependency for v1.0.** A demonstrably sound risk-scoring system is a stronger portfolio piece than a fragile chatbot layered on an unvalidated classifier.

## 5. High-level architecture

1. **Source data:** approved Kaggle CSV or clearly labeled synthetic fallback.
2. **Validation:** schema, missingness, duplicates, allowed ranges, leakage audit.
3. **Feature pipeline:** train-only imputers, encoders, scaling where appropriate.
4. **Training and evaluation:** baselines, tree ensemble, fixed random seeds, saved split definitions and metrics.
5. **Scoring:** model produces class probability or ranking score; optional separate Isolation Forest anomaly score.
6. **Explainability:** SHAP and model-metadata context.
7. **Presentation:** Streamlit overview, quality, performance and analyst queue pages.
8. **Optional serving:** FastAPI API routes, logs and Docker packaging.

**Important boundary:** The browser/dashboard is a decision-support UI. No decision automatically denies a claim or alleges fraud.

## 6. Technology and dependencies (names only)

| Layer | Tools | Reason |
|---|---|---|
| Core | Python 3.11+, Pandas, NumPy | Data pipelines and exploration |
| Analysis | JupyterLab, Matplotlib, Plotly | Reproducible notebooks and figures |
| Machine learning | scikit-learn, XGBoost | Baseline, ensemble, pipelines, metrics |
| Interpretation | SHAP | Model-level and claim-level explanations |
| Anomalies | scikit-learn IsolationForest | Independent outlier signal |
| Dashboard | Streamlit | Fast interactive demo |
| API (optional) | FastAPI, Pydantic, Uvicorn | Typed predictions / batch input |
| Storage (optional) | DuckDB / PostgreSQL | Analytical queries |
| Experiments | MLflow (optional) | Track runs and artifacts |
| Testing | pytest, Ruff | Quality and regression checks |
| Packaging | Docker, GitHub Actions | Reproducible deployment and CI |

Keep v1.0 dependencies lean; pin versions in an environment lock file when starting implementation.

## 7. Intended GitHub folder structure (conceptual, no implementation code)

- `README.md` — project overview, demo GIF/screenshots, reproducibility.
- `docs/` — specification, data dictionary, architecture, model card, ethics and limitations.
- `data/raw/` — gitignored external files.
- `data/processed/` — reproducible generated outputs, usually gitignored.
- `notebooks/` — exploration and baseline modeling notebooks.
- `src/data/` — loaders, validators and preprocessing.
- `src/features/` — feature definitions and transformations.
- `src/models/` — train, evaluate, score and save.
- `src/explainability/` — SHAP and interpretation.
- `src/anomaly/` — anomaly detection.
- `app/dashboard/` — Streamlit UI.
- `app/api/` — optional FastAPI service.
- `tests/` — test suite.
- `reports/figures/` — results plots and screenshots.
- `.github/workflows/` — continuous integration.
- `requirements` / dependency manifest, `.gitignore`, `LICENSE`.

## 8. Detailed four-week roadmap

### Week 1 — Data selection, quality and EDA

**Tasks**
- [ ] Open the primary Kaggle link; verify access and dataset usage terms.
- [ ] Download original `fraud_oracle.csv` and document version/date accessed.
- [ ] Inspect columns, target meaning, dataset origins and ethical caveats.
- [ ] Create data dictionary: feature, datatype, definition, permitted range, missingness and leakage risk.
- [ ] Examine positive class prevalence and distributions across key fields.
- [ ] Flag obvious target leakage and post-investigation information that would not exist at scoring time.
- [ ] Design charts for claim counts, severity, missingness and flag rates.
- [ ] Write a 1–2 page EDA findings summary with caveats.
- [ ] Create GitHub repository and initial README / project board.

**Deliverable:** validated dataset specification, EDA notebook/figures, documented target definition and project setup.

**Exit gate:** You can explain what every included model feature represents, and whether it would realistically be available when an analyst scores a claim.

### Week 2 — Supervised ML and evaluation

**Tasks**
- [ ] Define stratified train/validation/test splitting; keep test set untouched until final evaluation.
- [ ] Design train-only preprocessing to prevent leakage.
- [ ] Train Logistic Regression baseline.
- [ ] Train XGBoost or another tree ensemble.
- [ ] Address imbalance using class weights or resampling strictly inside training folds.
- [ ] Tune hyperparameters without test-set feedback.
- [ ] Evaluate PR-AUC, ROC-AUC, F1, precision, recall and confusion matrix.
- [ ] Plot precision-recall and calibration curves, and assess score usefulness at review budgets.
- [ ] Test threshold tradeoffs under explicit false-positive and missed-case assumptions.
- [ ] Save model artifacts and a concise evaluation report.

**Deliverable:** documented comparison table, one selected model, saved pipeline and evaluation figures.

**Exit gate:** The chosen model beats or appropriately contextualizes a baseline, and all numbers are explicitly marked as *held-out performance on the chosen historical dataset*.

### Week 3 — Explanations, anomalies and dashboard

**Tasks**
- [ ] Fit Isolation Forest separately and define anomaly flagging threshold.
- [ ] Add global feature importance and SHAP explanations for individual claims.
- [ ] Create overview page with operational summary metrics.
- [ ] Create model comparison / performance page.
- [ ] Create ranked investigation queue with configurable score threshold.
- [ ] Create claim detail with predicted score, anomaly score, feature contributions and limitations.
- [ ] Add downloadable CSV of scored cases.
- [ ] Test dashboard with sample cases and invalid/missing input values.

**Deliverable:** local Streamlit application with usable end-to-end analyst workflow.

**Exit gate:** A visitor can open the app, understand scores, review a sample claim and export cases without inspecting model source code.

### Week 4 — Reliability, demo and GitHub publication

**Tasks**
- [ ] Add tests for inputs, schema, stable output shape, and pipeline loading.
- [ ] Add reproducible dependency versions and dataset acquisition instructions.
- [ ] Add GitHub Actions automated checks.
- [ ] Optionally add a minimal FastAPI health and scoring endpoint.
- [ ] Add Docker packaging if deployment environment warrants it.
- [ ] Deploy a demo with no real personal claims information.
- [ ] Prepare screenshots, short demo video / GIF, architecture diagram and model card.
- [ ] Write limitations and responsible-use sections.
- [ ] Validate every README claim against real artifacts and metrics.
- [ ] Publish GitHub repository and pin it to profile.

**Deliverable:** public repo and demo with reproducible results.

**Exit gate:** A reviewer can reproduce key evaluation outputs, inspect model limitations and see the application running.

## 9. Measurement plan

**Predictive:** positive-class prevalence, PR-AUC (especially useful under class imbalance), ROC-AUC, precision, recall, F1, confusion matrix, Brier score/calibration. Include cross-validation variability when dataset is small.

**Operational:** precision@top 5/10/20% of claims, recall@K, fraction of queue selected, false flags per 100 reviewed cases, processing latency, batch upload success rate.

**Robustness:** evaluation by time (only if reliable timestamp allows), slice checks, missing-field sensitivity, model stability across random seeds, and leakage audit.

**Never invent success numbers.** Fill a results table only after completing and saving your actual runs.

## 10. Responsible AI, privacy and limitations

- Labels may be historical, noisy or biased and must never be represented as ground-truth legal determinations.
- Protected/demographic variables require special care; assess whether they should be excluded from training, and use group fairness analyses only when lawful and ethically appropriate.
- Do not store or log private customer data in the demo.
- Do not infer individual fraud from an anomaly flag or from SHAP feature contributions.
- Explain data provenance, labeling assumptions, training/testing strategy and dataset age.
- Avoid giving a 0–100 'fraud certainty' impression; call it a *model-estimated score for the dataset-defined flag*.
- Require human investigation; never authorize automatic claim denials.
- Check licensing and terms before uploading third-party dataset files to GitHub.

## 11. README and portfolio presentation checklist

- [ ] Clear one-sentence problem statement and product screenshot at top.
- [ ] 'Try the live demo' link when available.
- [ ] Dataset source link, version/date and data dictionary.
- [ ] Architecture diagram and reproducible run instructions.
- [ ] Baseline vs selected model comparison (real measured metrics only).
- [ ] Interactive sample investigation screenshots.
- [ ] Explainability and anomaly detection examples.
- [ ] Honest 'Limitations and responsible use' section.
- [ ] Testing/CI badges only once CI actually runs.
- [ ] Future roadmap and a 60–90 second screen recording.


## 12. Post-MVP improvements (optional)

1. Add a separate claims-severity regression module using the Allstate dataset, without mixing targets.
2. Add SQL-powered claim portfolio analytics in DuckDB/PostgreSQL.
3. Introduce drift dashboards, model version comparisons and MLflow tracking.
4. Add an evidence-grounded LLM assistant that describes precomputed findings with citations/traceability — never invent investigations or make decisions.
5. Add fairness and robustness audits, plus explicit model governance documentation.


---

### Sources and dataset pages (verified links, accessed October 2026)

- Vehicle insurance fraud dataset: https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection
- Alternative auto insurance claims dataset: https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data
- Allstate Claims Severity: https://www.kaggle.com/competitions/allstate-claims-severity/data
- CFPB Consumer Complaint Database: https://www.consumerfinance.gov/data-research/consumer-complaints/
- CFPB API release notes: https://cfpb.github.io/api/ccdb/release-notes.html


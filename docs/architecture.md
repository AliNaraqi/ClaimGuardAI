# Architecture

```mermaid
flowchart TD
    A[Kaggle CSV<br/>fraud_oracle.csv] -->|src/data/load.py<br/>schema validation| B[Validated raw DataFrame]
    B -->|src/features/build_features.py<br/>engineer_features: Age sentinel fix| C[Engineered DataFrame]
    C -->|src/models/train.py<br/>split_data: stratified 60/20/20| D1[Train]
    C --> D2[Validation]
    C --> D3[Test]

    D1 -->|ColumnTransformer fit on train only| E1[Logistic Regression]
    D1 --> E2[XGBoost]
    D2 -->|PR-AUC comparison| F[Model selection]
    E1 --> F
    E2 --> F
    F -->|selected_pipeline_xgboost.joblib| G[Saved model]

    G -->|scored once| D3
    D3 --> H[reports/model_results.json<br/>+ figures]

    G --> I[src/explainability/shap_utils.py<br/>TreeExplainer]
    D1 -->|unsupervised, no target| J[src/anomaly/isolation_forest.py]

    I --> K[data/processed/scored_test_claims.csv]
    J --> K

    K --> L[app/dashboard/app.py<br/>Streamlit: Overview, Data Quality,<br/>Model Performance, Investigation Queue]
    G --> M[app/api/main.py<br/>FastAPI: /health, /score]
    H --> L
```

## Boundary

The dashboard and API are decision-support surfaces only. Nothing in this architecture automatically denies a claim, accuses an individual, or writes back to any claims system — every scored output carries a human-review disclaimer (see [`docs/model_card.md`](model_card.md)).

## Layer responsibilities

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion | `src/data/` | Load CSV, validate schema against `src/data/schema.py` |
| Features | `src/features/` | Split-safe recoding (Age sentinel), train-only preprocessing pipeline |
| Modeling | `src/models/` | Stratified split, pipeline construction, evaluation metrics |
| Explainability | `src/explainability/` | SHAP global + per-claim local explanations |
| Anomaly | `src/anomaly/` | Unsupervised Isolation Forest, independent of the target |
| Presentation | `app/dashboard/` | Streamlit analyst UI |
| Serving (optional) | `app/api/` | FastAPI scoring endpoint for programmatic access |

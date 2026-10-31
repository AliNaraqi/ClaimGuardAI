# Architecture

![ClaimGuard AI architecture diagram](architecture_diagram.svg)

Interactive/source version (same content, Mermaid):

```mermaid
flowchart TD
    A["fraud_oracle.csv<br/>15,420 claims, 1994–96"] --> B["Prepare data<br/>validate · fix sentinels · build pipeline"]
    B --> C["Stratified split<br/>60% train · 20% val · 20% test"]

    C -->|train| E1["Logistic Regression<br/>baseline"]
    C -->|train| E2["XGBoost<br/>candidate"]
    E1 --> F["Model selection<br/>validation PR-AUC: 0.255 vs 0.154"]
    E2 --> F
    F --> G["Saved model<br/>selected_pipeline_xgboost.joblib"]

    C -.->|"train features only — never sees FraudFound_P"| J["Isolation Forest<br/>anomaly score"]

    G --> T["Test evaluation<br/>scored once · PR-AUC 0.242"]
    G --> I["SHAP<br/>explains predictions"]
    G --> M["FastAPI /score<br/>live, single-claim scoring"]

    I --> K["Scored claims<br/>risk + anomaly + SHAP reasons"]
    J --> K

    K --> L["Streamlit dashboard<br/>analyst queue + explanations"]
    T --> L
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

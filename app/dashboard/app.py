"""ClaimGuard AI — analyst-facing Streamlit dashboard.

Decision-support UI only: no page here denies a claim, alleges fraud, or takes any
automated action. Every score is presented as a model estimate requiring human review.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.schema import TARGET  # noqa: E402

st.set_page_config(page_title="ClaimGuard AI", layout="wide", page_icon=":shield:")

RAW_DATA_PATH = ROOT / "data" / "raw" / "fraud_oracle.csv"
SCORED_TEST_PATH = ROOT / "data" / "processed" / "scored_test_claims.csv"
MODEL_RESULTS_PATH = ROOT / "reports" / "model_results.json"
FIGURES_DIR = ROOT / "reports" / "figures"


@st.cache_data
def load_raw_data():
    if not RAW_DATA_PATH.exists():
        return None
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data
def load_scored_test():
    if not SCORED_TEST_PATH.exists():
        return None
    return pd.read_csv(SCORED_TEST_PATH)


@st.cache_data
def load_model_results():
    if not MODEL_RESULTS_PATH.exists():
        return None
    return json.loads(MODEL_RESULTS_PATH.read_text())


def limitations_banner():
    st.warning(
        "**Decision support only — not a fraud determination.** `FraudFound_P` is a "
        "historical (1994–1996), dataset-provided flag with uncertain provenance, not a "
        "verified legal finding. Risk and anomaly scores below require human review and "
        "must never be used to automatically deny a claim or accuse a specific person.",
        icon="⚠️",
    )


def parse_shap_reasons(reason_str: str) -> list[tuple[str, float]]:
    """top_shap_reasons is stored as 'feature=+0.12; feature2=-0.05; ...'."""
    if not isinstance(reason_str, str) or not reason_str.strip():
        return []
    parsed = []
    for part in reason_str.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        feat, val = part.rsplit("=", 1)
        try:
            parsed.append((feat.strip(), float(val)))
        except ValueError:
            continue
    return parsed


PAGES = ["Overview", "Data Quality", "Model Performance", "Investigation Queue"]

st.sidebar.title("ClaimGuard AI")
st.sidebar.caption("Explainable claims risk & decision intelligence — portfolio MVP")

# Deep-linkable pages (e.g. ?page=Model+Performance) so each view has a bookmarkable URL.
requested_page = st.query_params.get("page", PAGES[0])
default_index = PAGES.index(requested_page) if requested_page in PAGES else 0
page = st.sidebar.radio("Navigate", PAGES, index=default_index)
st.query_params["page"] = page
st.sidebar.divider()
st.sidebar.caption(
    "Demo mode: all claims shown are from the public, historical fraud_oracle dataset. "
    "No real customer data is used anywhere in this application."
)

limitations_banner()

raw_df = load_raw_data()

if raw_df is None:
    st.error(
        f"Raw dataset not found at `{RAW_DATA_PATH.relative_to(ROOT)}`. "
        "Download `fraud_oracle.csv` from Kaggle and place it there, then rerun."
    )
    st.stop()

# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total claims", f"{len(raw_df):,}")
    c2.metric("Flagged (FraudFound_P=1)", f"{int(raw_df[TARGET].sum()):,}")
    c3.metric("Flag prevalence", f"{raw_df[TARGET].mean():.2%}")
    c4.metric("Years covered", f"{raw_df['Year'].min()}–{raw_df['Year'].max()}")

    st.caption(
        "Prevalence reflects this historical dataset's own flagging process, not a "
        "current fraud rate estimate."
    )

    st.subheader("Claims by year")
    st.bar_chart(raw_df["Year"].value_counts().sort_index())

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Flag rate by vehicle category")
        g = raw_df.groupby("VehicleCategory")[TARGET].mean().sort_values(ascending=False)
        st.bar_chart(g)
    with col_b:
        st.subheader("Flag rate by base policy")
        g = raw_df.groupby("BasePolicy")[TARGET].mean().sort_values(ascending=False)
        st.bar_chart(g)

    st.caption(
        "These are descriptive associations within this historical, dataset-defined "
        "label only — not causal fraud risk factors."
    )

# ---------------------------------------------------------------------------
elif page == "Data Quality":
    st.title("Dataset Quality")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(raw_df):,}")
    c2.metric("Columns", f"{raw_df.shape[1]}")
    c3.metric("Duplicate rows", f"{int(raw_df.duplicated().sum())}")

    st.subheader("Known data-quality issues")
    age_zero = int((raw_df["Age"] == 0).sum())
    sentinel_rows = int((raw_df["DayOfWeekClaimed"] == "0").sum())
    st.markdown(
        f"""
- **{age_zero} rows** have `Age == 0`, which always co-occurs with
  `AgeOfPolicyHolder == '16 to 17'` — a known encoding quirk, not a literal age of zero.
- **{sentinel_rows} row(s)** carry a `'0'` sentinel in `DayOfWeekClaimed`/`MonthClaimed`
  instead of a real missing-value marker.
- `PolicyNumber` is a 1:1 row identifier ({raw_df['PolicyNumber'].nunique():,} unique
  values for {len(raw_df):,} rows) and is excluded from model features, along with
  `RepNumber`.
- No true NaN cells and no full-row duplicates were found — see
  [`docs/eda_findings.md`](../../docs/eda_findings.md) for the full audit.
"""
    )

    st.subheader("Column overview")
    dtype_table = pd.DataFrame(
        {
            "column": raw_df.columns,
            "dtype": raw_df.dtypes.astype(str).values,
            "n_unique": [raw_df[c].nunique() for c in raw_df.columns],
        }
    )
    st.dataframe(dtype_table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")

    results = load_model_results()
    if results is None:
        st.error(
            "No `reports/model_results.json` found. Run `notebooks/02_baseline_modeling.ipynb` "
            "first."
        )
        st.stop()

    st.caption(
        f"Selected model: **{results['selected_model']}** "
        f"(train={results['split']['train_n']}, val={results['split']['val_n']}, "
        f"test={results['split']['test_n']}, random_state={results['random_state']})"
    )

    st.subheader("Validation comparison (model selection)")
    val = results["validation_metrics"]
    comparison = pd.DataFrame(
        [
            {"model": "Logistic Regression", **{k: v for k, v in val["logistic_regression"].items() if k != "confusion_matrix"}},
            {"model": "XGBoost", **{k: v for k, v in val["xgboost"].items() if k != "confusion_matrix"}},
        ]
    ).set_index("model")
    st.dataframe(comparison, use_container_width=True)

    img = FIGURES_DIR / "pr_and_calibration_val.png"
    if img.exists():
        st.image(str(img), caption="Precision-recall and calibration curves (validation)")

    st.subheader(f"Held-out test evaluation — {results['selected_model']} (scored once)")
    test_metrics = results["test_metrics_selected_model"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{test_metrics['roc_auc']:.3f}")
    c2.metric("PR-AUC", f"{test_metrics['pr_auc']:.3f}")
    c3.metric("F1 @ 0.5", f"{test_metrics['f1']:.3f}")
    c4.metric("Brier score", f"{test_metrics['brier_score']:.3f}")

    cm_img = FIGURES_DIR / "test_confusion_matrix.png"
    if cm_img.exists():
        st.image(str(cm_img), caption="Test confusion matrix (threshold=0.5)", width=400)

    st.warning(
        "Calibration curve shows both models are **overconfident** — treat scores as a "
        "ranking signal, not a calibrated probability of fraud.",
        icon="📏",
    )

    st.subheader("Operational metrics: precision@K / recall@K")
    pr_at_k = pd.DataFrame(results["precision_recall_at_k_test"])
    st.dataframe(pr_at_k, use_container_width=True, hide_index=True)
    k_img = FIGURES_DIR / "recall_at_k_test.png"
    if k_img.exists():
        st.image(str(k_img), caption="Recall@K on held-out test set", width=500)

    st.subheader("Global feature importance (SHAP)")
    shap_img = FIGURES_DIR / "shap_global_importance.png"
    if shap_img.exists():
        st.image(str(shap_img))
        st.caption(
            "Known limitation: importance is partly dominated by rare vehicle-Make "
            "categories — see docs/explainability_and_anomaly.md before trusting "
            "Make-based explanations."
        )

    anomaly_img = FIGURES_DIR / "risk_vs_anomaly_scatter.png"
    if anomaly_img.exists():
        st.subheader("Risk score vs. anomaly score")
        st.image(str(anomaly_img))
        st.caption(
            "Isolation Forest anomaly score is an independent, unsupervised signal — "
            "not a fraud probability."
        )

# ---------------------------------------------------------------------------
elif page == "Investigation Queue":
    st.title("Investigation Queue")

    scored_df = load_scored_test()
    if scored_df is None:
        st.error(
            "No `data/processed/scored_test_claims.csv` found. Run "
            "`notebooks/03_explainability_and_anomaly.ipynb` first."
        )
        st.stop()

    st.caption(
        f"{len(scored_df):,} claims from the held-out test set, each carrying a "
        "supervised risk score and an independent anomaly score. This is demo data "
        "only — not a live production feed."
    )

    st.sidebar.subheader("Queue filters")
    min_risk = st.sidebar.slider(
        "Minimum risk score", 0.0, 1.0, float(st.query_params.get("min_risk", 0.0)), 0.01
    )
    outliers_only = st.sidebar.checkbox("Statistical outliers only (Isolation Forest)", value=False)
    vehicle_options = ["All"] + sorted(scored_df["VehicleCategory"].dropna().unique().tolist())
    vehicle_default = st.query_params.get("vehicle", "All")
    vehicle_filter = st.sidebar.selectbox(
        "Vehicle category",
        vehicle_options,
        index=vehicle_options.index(vehicle_default) if vehicle_default in vehicle_options else 0,
    )
    fault_options = ["All"] + sorted(scored_df["Fault"].dropna().unique().tolist())
    fault_filter = st.sidebar.selectbox("Fault", fault_options)
    st.query_params["min_risk"] = str(min_risk)
    st.query_params["vehicle"] = vehicle_filter

    filtered = scored_df[scored_df["risk_score"] >= min_risk]
    if outliers_only:
        filtered = filtered[filtered["is_statistical_outlier"]]
    if vehicle_filter != "All":
        filtered = filtered[filtered["VehicleCategory"] == vehicle_filter]
    if fault_filter != "All":
        filtered = filtered[filtered["Fault"] == fault_filter]
    filtered = filtered.sort_values("risk_score", ascending=False)

    st.metric("Claims in queue", f"{len(filtered):,} of {len(scored_df):,}")

    display_cols = [
        "PolicyNumber",
        "risk_score",
        "anomaly_score",
        "is_statistical_outlier",
        "Make",
        "VehicleCategory",
        "Fault",
        "BasePolicy",
        "AgeOfPolicyHolder",
        TARGET,
    ]
    st.dataframe(
        filtered[display_cols].rename(columns={TARGET: "FraudFound_P (dataset flag)"}),
        use_container_width=True,
        hide_index=True,
        height=350,
    )

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered queue as CSV",
        data=csv_bytes,
        file_name="claimguard_investigation_queue.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Claim detail")

    if len(filtered) == 0:
        st.info("No claims match the current filters.")
    else:
        policy_options = filtered["PolicyNumber"].tolist()
        selected_policy = st.selectbox("Select a claim (PolicyNumber)", policy_options)
        claim = filtered[filtered["PolicyNumber"] == selected_policy].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk score", f"{claim['risk_score']:.3f}")
        c2.metric("Anomaly score", f"{claim['anomaly_score']:.3f}")
        c3.metric(
            "Statistical outlier?",
            "Yes" if claim["is_statistical_outlier"] else "No",
        )

        st.markdown("**Top contributing features (local SHAP explanation):**")
        reasons = parse_shap_reasons(claim.get("top_shap_reasons", ""))
        if reasons:
            reason_df = pd.DataFrame(reasons, columns=["feature", "shap_value"])
            st.dataframe(reason_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No explanation available for this claim.")

        st.caption(
            "SHAP values explain this specific *model prediction* — they are not "
            "evidence of fraud, and vehicle-Make-based reasons carry a known "
            "reliability caveat (see Model Performance page)."
        )

        with st.expander("Full claim record"):
            st.dataframe(claim.to_frame().T, use_container_width=True, hide_index=True)

        st.info(
            "This claim requires human analyst review before any action is taken. "
            "No automated decision is made by this dashboard.",
            icon="🧑‍💼",
        )

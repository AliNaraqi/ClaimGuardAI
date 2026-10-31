"""Isolation Forest anomaly detection — a statistical outlier signal, independent of
the supervised fraud-flag model. An anomaly flag means "this claim's feature combination
is unusual relative to the training population," not "this claim is more likely fraud."
The two signals are kept and surfaced separately (see docs/explainability_and_anomaly.md).
"""

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from src.features.build_features import build_preprocessor

RANDOM_STATE = 42


def build_isolation_forest_pipeline(contamination: float = 0.06) -> Pipeline:
    """contamination is set near the dataset's own FraudFound_P prevalence (~6%) purely
    as a reporting convenience (roughly how many claims get flagged as outliers) — it is
    NOT a claim that outliers equal fraud; the two are evaluated as separate signals.
    """
    return Pipeline(
        [
            ("preprocess", build_preprocessor(scale_numeric=True)),
            (
                "model",
                IsolationForest(
                    n_estimators=300,
                    contamination=contamination,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def anomaly_scores(pipeline, X):
    """Higher score = more normal (sklearn convention for score_samples). We return a
    flipped, min-max-friendly 'anomaly_score' where HIGHER = more anomalous, which is
    more intuitive for an analyst-facing queue.
    """
    raw_score = pipeline.score_samples(X)  # higher = more normal
    anomaly_score = -raw_score  # higher = more anomalous
    is_anomaly = pipeline.predict(X) == -1  # sklearn: -1 outlier, 1 inlier
    return anomaly_score, is_anomaly

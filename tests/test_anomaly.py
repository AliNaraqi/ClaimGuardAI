from src.anomaly.isolation_forest import anomaly_scores, build_isolation_forest_pipeline
from src.models.train import split_data, xy


def test_isolation_forest_fits_unsupervised_and_scores(sample_df):
    train_df, _, test_df = split_data(sample_df)
    X_train, _ = xy(train_df)
    X_test, _ = xy(test_df)

    pipeline = build_isolation_forest_pipeline(contamination=0.1)
    pipeline.fit(X_train)  # no y passed: must work without the target

    scores, is_anomaly = anomaly_scores(pipeline, X_test)
    assert scores.shape[0] == len(X_test)
    assert is_anomaly.dtype == bool

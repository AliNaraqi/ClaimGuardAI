import joblib
import numpy as np

from src.explainability.shap_utils import compute_shap_values, explain_instance, global_importance
from src.features.build_features import get_feature_frame
from tests.conftest import requires_trained_model

MODEL_PATH = "models/selected_pipeline_xgboost.joblib"


@requires_trained_model
def test_compute_shap_values_and_importance(sample_df):
    X = get_feature_frame(sample_df)
    pipeline = joblib.load(MODEL_PATH)
    shap_values, dense, feature_names = compute_shap_values(pipeline, X.head(30))

    assert shap_values.shape == (30, len(feature_names))
    assert not np.isnan(shap_values).any()

    importance = global_importance(shap_values, feature_names, top_n=5)
    assert len(importance) == 5
    assert list(importance.columns) == ["feature", "mean_abs_shap"]

    local = explain_instance(shap_values[0], feature_names, dense[0], top_n=4)
    assert len(local) == 4
    assert set(local.columns) == {"feature", "feature_value", "shap_value"}

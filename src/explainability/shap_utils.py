"""SHAP-based explanations for the trained risk model.

Important: SHAP values explain the MODEL's prediction (which features pushed this
specific prediction up or down relative to the model's average output). They do not
establish that a feature caused fraud, and they are not evidence of wrongdoing.
"""

import numpy as np
import pandas as pd
import shap


def get_feature_names(preprocessor) -> list:
    """Feature names after the ColumnTransformer's numeric + one-hot categorical steps."""
    numeric_names = preprocessor.transformers_[0][2]
    cat_pipeline = preprocessor.named_transformers_["categorical"]
    cat_cols = preprocessor.transformers_[1][2]
    onehot = cat_pipeline.named_steps["onehot"]
    cat_names = onehot.get_feature_names_out(cat_cols).tolist()
    return list(numeric_names) + cat_names


def build_explainer(pipeline):
    """TreeExplainer on the fitted model step; operates on the preprocessed feature space."""
    model = pipeline.named_steps["model"]
    return shap.TreeExplainer(model)


def compute_shap_values(pipeline, X: pd.DataFrame):
    """Returns (shap_values ndarray [n_samples, n_features], transformed_dense ndarray,
    feature_names list) for claims in X.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    transformed = preprocessor.transform(X)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    feature_names = get_feature_names(preprocessor)

    explainer = build_explainer(pipeline)
    shap_values = explainer.shap_values(dense)
    if isinstance(shap_values, list):  # some SHAP/XGBoost versions return [class0, class1]
        shap_values = shap_values[1]
    return shap_values, dense, feature_names


def global_importance(shap_values, feature_names, top_n: int = 20) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    out = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return out.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def explain_instance(shap_values_row, feature_names, dense_row, top_n: int = 8) -> pd.DataFrame:
    """Top contributing features for one already-computed SHAP row."""
    out = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_value": dense_row,
            "shap_value": shap_values_row,
        }
    )
    out["abs_shap"] = out["shap_value"].abs()
    return out.sort_values("abs_shap", ascending=False).head(top_n).drop(columns="abs_shap").reset_index(drop=True)

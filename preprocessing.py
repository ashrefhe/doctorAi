
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessor(X: pd.DataFrame, numeric_cols: list, categorical_cols: list):
    """
    Build (but do NOT fit) a ColumnTransformer preprocessor.
    Fitting must happen only inside CV / train folds to avoid data leakage.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def auto_preprocess(df: pd.DataFrame, target_col: str, task: str):
    """
    Prepare X/y WITHOUT fitting any transformer globally.
    Returns:
      X_raw, y_encoded_or_float, preprocessor, label_encoder,
      numeric_cols, categorical_cols,
      class_dist_before, imbalance_ratio, imbalance_detected, high_card
    """
    df = df.copy()
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # Column types
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Drop high-cardinality categoricals
    high_card = [c for c in categorical_cols if X[c].nunique(dropna=True) > 50]
    categorical_cols = [c for c in categorical_cols if c not in high_card]

    preprocessor = build_preprocessor(X, numeric_cols, categorical_cols)

    label_encoder = None
    class_dist_before = {}
    imbalance_ratio = 1.0
    imbalance_detected = False

    if task == "classification":
        y_str = y_raw.astype(str)
        class_dist_before = y_str.value_counts(dropna=False).to_dict()

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_str)

        if class_dist_before:
            counts = np.array(list(class_dist_before.values()), dtype=float)
            min_count = counts.min()
            max_count = counts.max()
            imbalance_ratio = float(max_count / max(min_count, 1.0))
            imbalance_detected = bool(len(counts) > 1 and imbalance_ratio > 1.5)
    else:
        y = y_raw.values.astype(float)

    return (
        X,
        y,
        preprocessor,
        label_encoder,
        numeric_cols,
        categorical_cols,
        class_dist_before,
        imbalance_ratio,
        imbalance_detected,
        high_card,
    )


def get_preprocessing_summary(
    numeric_cols,
    categorical_cols,
    df,
    target_col,
    class_dist_before=None,
    imbalance_ratio=1.0,
    imbalance_detected=False,
    high_card=None,
    resampling_info=None,
):
    missing_total = int(df.drop(columns=[target_col]).isnull().sum().sum())
    return {
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols),
        "total_missing_values": missing_total,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "class_dist_before": class_dist_before or {},
        "imbalance_ratio": float(imbalance_ratio),
        "imbalance_detected": bool(imbalance_detected),
        "high_card_dropped": high_card or [],
        "resampling_info": resampling_info or {},
    }

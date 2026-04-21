import pandas as pd
import numpy as np


def infer_task(df: pd.DataFrame, target_col: str) -> str:
    """
    Infer whether the ML task is classification or regression
    based on the target column's properties.
    """
    target = df[target_col].dropna()

    if target.dtype == object or str(target.dtype) == "category":
        return "classification"

    if target.dtype == bool:
        return "classification"

    n_unique = target.nunique()
    n_total = len(target)

    if n_unique <= 20 and n_unique / n_total < 0.05:
        return "classification"

    if pd.api.types.is_integer_dtype(target) and n_unique <= 15:
        return "classification"

    return "regression"


def is_ambiguous_task(df: pd.DataFrame, target_col: str) -> bool:
    """
    Returns True when the auto-detection result is heuristic and uncertain.
    Used to show a warning in the UI so the user knows they can override.
    """
    target = df[target_col].dropna()

    # Clear-cut cases: not ambiguous
    if target.dtype == object or str(target.dtype) == "category" or target.dtype == bool:
        return False

    n_unique = target.nunique()
    n_total = len(target)

    # Ambiguous zone: numeric integer with small-to-medium unique count
    if pd.api.types.is_integer_dtype(target) and 10 < n_unique <= 30:
        return True

    # Ambiguous zone: float but behaves discretely
    if pd.api.types.is_float_dtype(target):
        frac = (target % 1 == 0).mean()
        if frac > 0.95 and n_unique <= 30:
            return True

    # Ambiguous zone: borderline ratio
    if 10 < n_unique <= 25 and 0.03 < n_unique / n_total < 0.10:
        return True

    return False


def get_task_explanation(task: str, target_col: str, df: pd.DataFrame) -> str:
    target = df[target_col].dropna()
    n_unique = target.nunique()

    if task == "classification":
        classes = target.unique()
        return (
            f"**Classification** detected: target '{target_col}' has {n_unique} unique classes "
            f"({list(classes[:5])}{'...' if n_unique > 5 else ''})."
        )
    else:
        return (
            f"**Regression** detected: target '{target_col}' is continuous "
            f"(range: {target.min():.4f} → {target.max():.4f}, mean: {target.mean():.4f})."
        )

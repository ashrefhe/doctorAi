"""
ml_engine.py — Orchestrates the full ML pipeline:
  1. Task inference / override
  2. Train / test holdout split (20 %) — effectué AVANT tout fitting
  3. Preprocessing (raw X + unfitted preprocessor — aucune fuite)
  4. Model selection
  5. Grid search + cross-validation (Pipeline par modèle, sur train seulement)
  6. Évaluation finale sur le holdout test
  7. SHAP values pour le meilleur modèle
  8. Charts
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import auto_preprocess, get_preprocessing_summary
from task_inference import infer_task, get_task_explanation
from model_selection import get_models_and_params
from evaluation import run_grid_search, build_results_charts


def _compute_shap(best_estimator, X_test_raw, feature_names: list, task: str) -> dict:
    try:
        import shap
        pipe = best_estimator
        step_names = [s[0] for s in pipe.steps]

        if "preprocessor" in step_names:
            preprocessor_step = pipe.named_steps["preprocessor"]
            X_test_transformed = preprocessor_step.transform(X_test_raw)
        else:
            from sklearn.pipeline import Pipeline as SkPipeline
            pre_steps = pipe.steps[:-1]
            pre_pipe = SkPipeline(pre_steps)
            X_test_transformed = pre_pipe.transform(X_test_raw)

        model_step = pipe.steps[-1][1]

        try:
            ohe_names = preprocessor_step.get_feature_names_out()
        except Exception:
            ohe_names = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]

        model_class = type(model_step).__name__
        tree_models = {
            "RandomForestClassifier", "RandomForestRegressor",
            "GradientBoostingClassifier", "GradientBoostingRegressor",
            "XGBClassifier", "XGBRegressor",
            "LGBMClassifier", "LGBMRegressor",
            "DecisionTreeClassifier", "DecisionTreeRegressor",
        }

        sample_size = min(200, X_test_transformed.shape[0])
        X_sample = X_test_transformed[:sample_size]

        if model_class in tree_models:
            explainer = shap.TreeExplainer(model_step)
            shap_values = explainer.shap_values(X_sample)
        else:
            background = shap.kmeans(X_test_transformed, min(50, X_test_transformed.shape[0]))
            explainer = shap.KernelExplainer(model_step.predict, background)
            shap_values = explainer.shap_values(X_sample, nsamples=100)

        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                sv = np.abs(shap_values[1])
            else:
                sv = np.mean([np.abs(s) for s in shap_values], axis=0)
        else:
            sv = np.abs(shap_values)

        mean_abs = np.mean(sv, axis=0)
        top_n = min(20, len(mean_abs))
        sorted_idx = np.argsort(mean_abs)[::-1][:top_n]

        return {
            "feature_names": [str(ohe_names[i]) for i in sorted_idx],
            "mean_abs_shap": [float(mean_abs[i]) for i in sorted_idx],
            "available": True,
        }

    except Exception as e:
        return {
            "feature_names": [],
            "mean_abs_shap": [],
            "available": False,
            "error": str(e),
        }


def run_pipeline(
    df: pd.DataFrame,
    target_col: str,
    cv_folds: int = 5,
    task_override: str = None,
    mode: str = "fast",
    test_size: float = 0.2,
):
    if task_override and task_override in ("classification", "regression"):
        task = task_override
    else:
        task = infer_task(df, target_col)

    task_explanation = get_task_explanation(task, target_col, df)

    (
        X_raw,
        y,
        preprocessor,
        label_encoder,
        numeric_cols,
        categorical_cols,
        class_dist_before,
        imbalance_ratio,
        imbalance_detected,
        high_card,
    ) = auto_preprocess(df, target_col, task)

    stratify = y if task == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y, test_size=test_size, random_state=42, stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y, test_size=test_size, random_state=42,
        )

    models_params = get_models_and_params(task, mode=mode)
    results, resampling_info = run_grid_search(
        models_params, X_train, y_train, preprocessor, task, cv=cv_folds,
    )

    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        r2_score, mean_absolute_error, mean_squared_error,
    )

    for r in results:
        if r.get("estimator") is None:
            r["holdout"] = {}
            continue
        try:
            y_pred = r["estimator"].predict(X_test)
            if task == "classification":
                holdout = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                }
                if len(np.unique(y)) == 2:
                    try:
                        y_prob = r["estimator"].predict_proba(X_test)[:, 1]
                        holdout["roc_auc"] = float(roc_auc_score(y_test, y_prob))
                    except Exception:
                        pass
            else:
                holdout = {
                    "r2": float(r2_score(y_test, y_pred)),
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                }
            r["holdout"] = holdout
        except Exception as e:
            r["holdout"] = {"error": str(e)}

    best = results[0]
    shap_data = {}
    if best.get("estimator") is not None:
        shap_data = _compute_shap(best["estimator"], X_test, numeric_cols + categorical_cols, task)

    bar_fig, violin_fig, radar_fig = build_results_charts(results, task)

    prep_summary = get_preprocessing_summary(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        df=df,
        target_col=target_col,
        class_dist_before=class_dist_before,
        imbalance_ratio=imbalance_ratio,
        imbalance_detected=imbalance_detected,
        high_card=high_card,
        resampling_info=resampling_info,
    )
    prep_summary["holdout_test_size"] = len(y_test)
    prep_summary["train_size"] = len(y_train)

    return {
        "task": task,
        "task_explanation": task_explanation,
        "preprocessing": prep_summary,
        "results": results,
        "best_model": best,
        "bar_chart": bar_fig,
        "violin_chart": violin_fig,
        "radar_chart": radar_fig,
        "label_encoder": label_encoder,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "cv_folds": cv_folds,
        "resampling_info": resampling_info,
        "shap_data": shap_data,
        "X_test": X_test,
        "y_test": y_test,
        "test_size": test_size,
    }
